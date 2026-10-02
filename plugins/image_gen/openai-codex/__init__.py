"""OpenAI image generation backend — ChatGPT/Codex OAuth variant.

Identical model catalog and tier semantics to the ``openai`` image-gen plugin
(``gpt-image-2`` at low/medium/high quality), but routes the request through
the Codex Responses API ``image_generation`` tool instead of the
``images.generate`` REST endpoint. This lets users who are already
authenticated with Codex/ChatGPT generate images without configuring a
separate ``OPENAI_API_KEY``.

Selection precedence for the tier (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai-codex.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``

The host chat model that calls the ``image_generation`` tool is resolved at
runtime from the raw authenticated Codex ``/models`` catalog (see
:func:`_resolve_host_models`). ``image_gen.openai-codex.host_model`` in
``config.yaml`` pins a specific host, but only when that slug is present in
the account's raw catalog.

Output is saved as PNG under ``$HERMES_HOME/cache/images/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog — mirrors the ``openai`` plugin so the picker UX is identical.
# ---------------------------------------------------------------------------

API_MODEL = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

# Codex Responses surface used for the request. The chat model is only the
# host that calls the ``image_generation`` tool; the actual image work is done
# by ``API_MODEL``. The host is discovered per account because the Codex
# backend rejects slugs outside the account's lineup with HTTP 400.
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_MODELS_URL = f"{_CODEX_BASE_URL}/models?client_version=1.0.0"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation requests by "
    "using the image_generation tool when provided."
)

# Mainline host slugs look like ``gpt-6``, ``gpt-5.5`` or ``gpt-6-sol``.
# Specialised variants (codex / mini / nano / pro / spark / multi-suffix) are
# skipped — they are not general-purpose tool hosts.
_MAINLINE_HOST_RE = re.compile(r"^gpt-\d+(?:\.\d+)*(?:-([a-z0-9]+))?$")
_NON_MAINLINE_SUFFIXES = frozenset({"codex", "mini", "nano", "pro", "spark", "oss"})
_MAX_HOST_ATTEMPTS = 3

# In-process cache of the raw /models catalog, keyed by a token digest so the
# OAuth token itself is never held as a key. Failures are never cached.
_HOST_MODEL_CACHE_TTL = 600.0
_HOST_MODEL_CACHE: Dict[str, Tuple[float, List[str]]] = {}


class CodexHostModelError(RuntimeError):
    """No usable Codex host model could be determined for image generation."""


class CodexHostModelRejected(RuntimeError):
    """The Codex backend refused the chosen host model for this account."""

    def __init__(self, host_model: str, detail: str) -> None:
        super().__init__(f"Codex rejected host model '{host_model}': {detail}")
        self.host_model = host_model


# ---------------------------------------------------------------------------
# Config + auth helpers
# ---------------------------------------------------------------------------


def _load_image_gen_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    import os

    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_image_gen_config()
    sub = cfg.get("openai-codex") if isinstance(cfg.get("openai-codex"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(sub, dict):
        value = sub.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _read_codex_access_token() -> Optional[str]:
    """Return a usable Codex OAuth token, or None.

    Delegates to the canonical reader in ``agent.auxiliary_client`` so token
    expiry, credential pool selection, and JWT decoding stay in one place.
    """
    try:
        from agent.auxiliary_client import _read_codex_access_token as _reader

        token = _reader()
        if isinstance(token, str) and token.strip():
            return token.strip()
        return None
    except Exception as exc:
        logger.debug("Could not resolve Codex access token: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Host model discovery
# ---------------------------------------------------------------------------


_DEFAULT_PRIORITY_RANK = 10_000


def _priority_rank(priority: Any) -> int:
    """Map a /models ``priority`` to a sort rank; unusable values sort last."""
    if isinstance(priority, bool):
        return _DEFAULT_PRIORITY_RANK
    if isinstance(priority, int):
        return priority
    if isinstance(priority, float) and math.isfinite(priority):
        return int(priority)
    return _DEFAULT_PRIORITY_RANK


def _fetch_raw_codex_models(token: str) -> List[str]:
    """Return visible slugs from the raw authenticated Codex ``/models`` catalog.

    Deliberately bypasses ``hermes_cli.codex_models``: that helper appends
    synthetic forward-compat slugs and hardcoded fallbacks the account may not
    be able to use. Raises :class:`CodexHostModelError` on any failure.
    """
    import httpx
    from agent.auxiliary_client import _codex_cloudflare_headers

    headers = _codex_cloudflare_headers(token)
    headers["Authorization"] = f"Bearer {token}"
    try:
        resp = httpx.get(_CODEX_MODELS_URL, headers=headers, timeout=15.0)
    except Exception as exc:
        # Only the exception type: the message (and chained traceback) can
        # echo request headers, including the bearer token.
        raise CodexHostModelError(
            f"Codex /models request failed: {type(exc).__name__}"
        ) from None
    if resp.status_code != 200:
        raise CodexHostModelError(f"Codex /models returned HTTP {resp.status_code}")
    try:
        data = resp.json()
    except Exception as exc:
        raise CodexHostModelError("Codex /models returned a non-JSON body") from exc
    entries = data.get("models") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise CodexHostModelError("Codex /models response has no 'models' list")

    sortable: List[Tuple[int, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        visibility = item.get("visibility")
        if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
            continue
        sortable.append((_priority_rank(item.get("priority")), slug.strip()))

    sortable.sort(key=lambda entry: (entry[0], entry[1]))
    slugs: List[str] = []
    for _, slug in sortable:
        if slug not in slugs:
            slugs.append(slug)
    return slugs


def _cached_raw_codex_models(token: str) -> List[str]:
    """:func:`_fetch_raw_codex_models` with a short per-token in-process cache."""
    key = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = time.monotonic()
    hit = _HOST_MODEL_CACHE.get(key)
    if hit and now - hit[0] < _HOST_MODEL_CACHE_TTL:
        return list(hit[1])
    slugs = _fetch_raw_codex_models(token)
    if slugs:
        _HOST_MODEL_CACHE[key] = (now, list(slugs))
    return slugs


def _rank_host_models(slugs: List[str]) -> List[str]:
    """Keep mainline host candidates, preserving the backend's priority order."""
    ranked: List[str] = []
    for slug in slugs:
        match = _MAINLINE_HOST_RE.match(slug)
        if match and match.group(1) not in _NON_MAINLINE_SUFFIXES and slug not in ranked:
            ranked.append(slug)
    return ranked


def _configured_host_model() -> Optional[str]:
    sub = _load_image_gen_config().get("openai-codex")
    value = sub.get("host_model") if isinstance(sub, dict) else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _resolve_host_models(token: str) -> List[str]:
    """Return host model candidates (best first) for this Codex account.

    A configured ``image_gen.openai-codex.host_model`` is honoured only when
    the raw catalog lists it. Raises :class:`CodexHostModelError` instead of
    guessing a hardcoded slug when discovery fails or finds nothing usable.
    """
    slugs = _cached_raw_codex_models(token)
    if not slugs:
        raise CodexHostModelError("Codex /models returned no models for this account")

    override = _configured_host_model()
    if override:
        if override in slugs:
            return [override]
        raise CodexHostModelError(
            f"Configured image_gen.openai-codex.host_model '{override}' is not in "
            f"this account's Codex model list: {', '.join(slugs)}"
        )

    ranked = _rank_host_models(slugs)
    if not ranked:
        raise CodexHostModelError(
            "No mainline Codex host model found in this account's model list "
            f"({', '.join(slugs)}); set image_gen.openai-codex.host_model to one of them"
        )
    return ranked


def _build_responses_payload(
    *, prompt: str, size: str, quality: str, host_model: str
) -> Dict[str, Any]:
    """Build the Codex Responses request body for an image_generation call."""
    return {
        "model": host_model,
        "store": False,
        "instructions": _CODEX_INSTRUCTIONS,
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }],
        "tools": [{
            "type": "image_generation",
            "model": API_MODEL,
            "size": size,
            "quality": quality,
            "output_format": "png",
            "background": "opaque",
            "partial_images": 1,
        }],
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "image_generation"}],
        },
        "stream": True,
    }


def _extract_image_b64(value: Any) -> Optional[str]:
    """Return the newest image b64 embedded in a Responses event payload."""
    found: Optional[str] = None
    if isinstance(value, dict):
        if value.get("type") == "image_generation_call":
            result = value.get("result")
            if isinstance(result, str) and result:
                found = result
        partial = value.get("partial_image_b64")
        if isinstance(partial, str) and partial:
            found = partial
        for child in value.values():
            nested = _extract_image_b64(child)
            if nested:
                found = nested
    elif isinstance(value, list):
        for child in value:
            nested = _extract_image_b64(child)
            if nested:
                found = nested
    return found


def _iter_sse_json(response: Any):
    """Yield JSON payloads from an SSE response without OpenAI SDK parsing.

    The ChatGPT/Codex backend can emit image-generation events newer than the
    pinned Python SDK understands. Parsing raw SSE keeps this provider tolerant
    of those event-shape changes.
    """
    event_name: Optional[str] = None
    data_lines: List[str] = []

    def flush():
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        raw = "\n".join(data_lines).strip()
        event = event_name
        event_name = None
        data_lines = []
        if not raw or raw == "[DONE]":
            return None
        payload = json.loads(raw)
        if isinstance(payload, dict) and event and "type" not in payload:
            payload["type"] = event
        return payload

    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = str(line)
        if line == "":
            payload = flush()
            if payload is not None:
                yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())

    payload = flush()
    if payload is not None:
        yield payload


def _collect_image_b64(
    token: str, *, prompt: str, size: str, quality: str, host_model: str
) -> Optional[str]:
    """Stream a Codex Responses image_generation call and return the b64 image."""
    import httpx
    from agent.auxiliary_client import _codex_cloudflare_headers

    headers = _codex_cloudflare_headers(token)
    headers.update({
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    payload = _build_responses_payload(
        prompt=prompt, size=size, quality=quality, host_model=host_model
    )
    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=30.0, pool=30.0)

    image_b64: Optional[str] = None
    with httpx.Client(timeout=timeout, headers=headers) as http:
        with http.stream("POST", f"{_CODEX_BASE_URL}/responses", json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                exc.response.read()
                body = exc.response.text[:500]
                if exc.response.status_code == 400 and "not supported" in body.lower():
                    raise CodexHostModelRejected(host_model, body[:200]) from exc
                raise RuntimeError(
                    f"Codex Responses API returned HTTP {exc.response.status_code}: {body}"
                ) from exc
            for event in _iter_sse_json(response):
                found = _extract_image_b64(event)
                if found:
                    image_b64 = found

    return image_b64


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAICodexImageGenProvider(ImageGenProvider):
    """gpt-image-2 routed through ChatGPT/Codex OAuth instead of an API key."""

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI (Codex auth)"

    def is_available(self) -> bool:
        if not _read_codex_access_token():
            return False
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI (Codex auth)",
            "badge": "free",
            "tag": "gpt-image-2 via ChatGPT/Codex OAuth — no API key required",
            "env_vars": [],
            "post_setup_hint": (
                "Sign in with `hermes auth codex` (or `hermes setup` → Codex) "
                "if you haven't already. No API key needed."
            ),
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        if not _read_codex_access_token():
            return error_response(
                error=(
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                ),
                error_type="auth_required",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        try:
            import httpx  # noqa: F401
        except ImportError:
            return error_response(
                error="httpx Python package not installed (pip install httpx)",
                error_type="missing_dependency",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        token = _read_codex_access_token()
        if not token:
            return error_response(
                error=(
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                ),
                error_type="auth_required",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            host_models = _resolve_host_models(token)
        except CodexHostModelError as exc:
            return error_response(
                error=f"Could not select a Codex host model for image generation: {exc}",
                error_type="host_model_unavailable",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        b64: Optional[str] = None
        host_model = host_models[0]
        rejected: List[str] = []
        try:
            for host_model in host_models[:_MAX_HOST_ATTEMPTS]:
                try:
                    b64 = _collect_image_b64(
                        token,
                        prompt=prompt,
                        size=size,
                        quality=meta["quality"],
                        host_model=host_model,
                    )
                    break
                except CodexHostModelRejected:
                    logger.info("Codex rejected image host model %s; trying next", host_model)
                    rejected.append(host_model)
            else:
                return error_response(
                    error=(
                        "Codex rejected every candidate host model for image "
                        f"generation ({', '.join(rejected)}); set "
                        "image_gen.openai-codex.host_model to a supported model"
                    ),
                    error_type="host_model_unavailable",
                    provider="openai-codex",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        except Exception as exc:
            logger.debug("Codex image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation via Codex auth failed: {exc}",
                error_type="api_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not b64:
            return error_response(
                error="Codex response contained no image_generation_call result",
                error_type="empty_response",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(b64, prefix=f"openai_codex_{tier_id}")
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai-codex",
            extra={"size": size, "quality": meta["quality"], "host_model": host_model},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())
