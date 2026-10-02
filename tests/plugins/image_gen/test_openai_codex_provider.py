"""Tests for the bundled ``openai-codex`` image_gen plugin.

Mirrors ``test_openai_provider.py`` but targets the standalone
Codex/ChatGPT-OAuth-backed provider that uses the Responses
``image_generation`` tool path instead of the ``images.generate`` REST
endpoint.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# The plugin directory uses a hyphen, which is not a valid Python identifier
# for the dotted-import form. Load it via importlib so tests don't need to
# touch sys.path or rename the directory.
codex_plugin = importlib.import_module("plugins.image_gen.openai-codex")


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture(autouse=True)
def _clear_host_model_cache():
    cache = getattr(codex_plugin, "_HOST_MODEL_CACHE", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


@pytest.fixture
def provider(monkeypatch):
    # Codex plugin is API-key-independent; clear it to make the test honest.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return codex_plugin.OpenAICodexImageGenProvider()


# Raw authenticated /models slugs observed on a ChatGPT Plus account.
_PLUS_CATALOG = [
    "gpt-6.1-sol",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
]


@pytest.fixture
def plus_catalog(monkeypatch):
    calls = []

    def _fetch(token):
        calls.append(token)
        return list(_PLUS_CATALOG)

    monkeypatch.setattr(codex_plugin, "_fetch_raw_codex_models", _fetch, raising=False)
    return calls


def _set_host_override(monkeypatch, value):
    monkeypatch.setattr(
        codex_plugin,
        "_load_image_gen_config",
        lambda: {"openai-codex": {"host_model": value}},
    )


class _FakeModelsResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "openai-codex"

    def test_display_name(self, provider):
        assert provider.display_name == "OpenAI (Codex auth)"

    def test_default_model(self, provider):
        assert provider.default_model() == "gpt-image-2-medium"

    def test_list_models_three_tiers(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert ids == ["gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high"]

    def test_setup_schema_has_no_required_env_vars(self, provider):
        schema = provider.get_setup_schema()
        assert schema["env_vars"] == []
        assert schema["badge"] == "free"


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_unavailable_without_codex_token(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False

    def test_available_with_codex_token(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is True

    def test_openai_api_key_alone_is_not_enough(self, monkeypatch):
        # Codex plugin is intentionally orthogonal to the API-key plugin —
        # the API key alone must NOT make it appear available.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False


# ── Generate ────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_returns_auth_error_without_codex_token(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: None)
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_returns_invalid_argument_for_empty_prompt(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        result = provider.generate("   ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_generate_uses_codex_stream_path(self, provider, monkeypatch, tmp_path, plus_catalog):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", lambda *a, **kw: _b64_png())

        result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["provider"] == "openai-codex"
        assert result["quality"] == "medium"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        # Filename prefix differs from the API-key plugin so cache audits can
        # tell the two backends apart.
        assert saved.name.startswith("openai_codex_")

    def test_codex_stream_request_shape(self, provider, monkeypatch, plus_catalog):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        captured = {}

        def _collect(token, *, prompt, size, quality, host_model):
            captured.update(codex_plugin._build_responses_payload(
                prompt=prompt,
                size=size,
                quality=quality,
                host_model=host_model,
            ))
            return _b64_png()

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _collect)

        result = provider.generate("a cat", aspect_ratio="portrait")
        assert result["success"] is True
        assert result["host_model"] == "gpt-6.1-sol"

        # Host model comes from the raw catalog, not a hardcoded slug.
        assert captured["model"] == "gpt-6.1-sol"
        assert captured["store"] is False
        assert captured["input"][0]["type"] == "message"
        assert captured["input"][0]["role"] == "user"
        assert captured["input"][0]["content"][0]["type"] == "input_text"
        assert captured["tool_choice"]["type"] == "allowed_tools"
        assert captured["tool_choice"]["mode"] == "required"
        assert captured["tool_choice"]["tools"] == [{"type": "image_generation"}]

        tool = captured["tools"][0]
        assert tool["type"] == "image_generation"
        assert tool["model"] == "gpt-image-2"
        assert tool["quality"] == "medium"
        assert tool["size"] == "1024x1536"
        assert tool["output_format"] == "png"
        assert tool["background"] == "opaque"
        assert tool["partial_images"] == 1

    def test_partial_image_event_used_when_done_missing(self):
        """If output_item.done is missing, partial_image_b64 is accepted."""
        payload = {
            "type": "response.image_generation_call.partial_image",
            "partial_image_b64": _b64_png(),
        }
        assert codex_plugin._extract_image_b64(payload) == _b64_png()

    def test_sse_parser_handles_event_and_data_lines(self):
        class _Response:
            def iter_lines(self):
                return iter([
                    "event: response.output_item.done",
                    'data: {"item": {"type": "image_generation_call", "result": "abc"}}',
                    "",
                ])

        events = list(codex_plugin._iter_sse_json(_Response()))
        assert events == [{
            "type": "response.output_item.done",
            "item": {"type": "image_generation_call", "result": "abc"},
        }]

    def test_final_response_sweep_recovers_image(self):
        """Completed response output is found by recursive payload scanning."""
        payload = {
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "image_generation_call",
                    "status": "completed",
                    "id": "ig_final",
                    "result": _b64_png(),
                }],
            },
        }
        assert codex_plugin._extract_image_b64(payload) == _b64_png()

    def test_empty_response_returns_error(self, provider, monkeypatch, plus_catalog):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        monkeypatch.setattr(codex_plugin, "_collect_image_b64", lambda *a, **kw: None)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_stream_exception_returns_api_error(self, provider, monkeypatch, plus_catalog):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        def _boom(*args, **kwargs):
            raise RuntimeError("cloudflare 403")

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _boom)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "cloudflare 403" in result["error"]

    def test_payload_uses_configured_host_override(self, provider, monkeypatch, plus_catalog):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        _set_host_override(monkeypatch, "gpt-6-sol")
        hosts = []

        def _collect(token, *, prompt, size, quality, host_model):
            hosts.append(host_model)
            return _b64_png()

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _collect)

        result = provider.generate("a cat")
        assert result["success"] is True
        assert hosts == ["gpt-6-sol"]
        # Tier selection is independent of host selection.
        assert result["model"] == "gpt-image-2-medium"
        assert result["quality"] == "medium"

    def test_discovery_failure_fails_without_stale_fallback(self, provider, monkeypatch):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")

        def _fetch(token):
            raise codex_plugin.CodexHostModelError("Codex /models returned HTTP 503")

        monkeypatch.setattr(codex_plugin, "_fetch_raw_codex_models", _fetch, raising=False)
        hosts = []
        monkeypatch.setattr(
            codex_plugin,
            "_collect_image_b64",
            lambda *a, **kw: hosts.append(kw.get("host_model")) or _b64_png(),
        )

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "host_model_unavailable"
        assert "HTTP 503" in result["error"]
        assert hosts == []

    def test_unsupported_host_falls_through_to_next_candidate(
        self, provider, monkeypatch, plus_catalog
    ):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        hosts = []

        def _collect(token, *, prompt, size, quality, host_model):
            hosts.append(host_model)
            if host_model == "gpt-6.1-sol":
                raise codex_plugin.CodexHostModelRejected(host_model, "not supported")
            return _b64_png()

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _collect)

        result = provider.generate("a cat")
        assert result["success"] is True
        assert hosts == ["gpt-6.1-sol", "gpt-6-astra"]
        assert result["host_model"] == "gpt-6-astra"

    def test_rejected_override_is_not_replaced(self, provider, monkeypatch, plus_catalog):
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
        _set_host_override(monkeypatch, "gpt-6-sol")
        hosts = []

        def _collect(token, *, prompt, size, quality, host_model):
            hosts.append(host_model)
            raise codex_plugin.CodexHostModelRejected(host_model, "not supported")

        monkeypatch.setattr(codex_plugin, "_collect_image_b64", _collect)

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "host_model_unavailable"
        assert "gpt-6-sol" in result["error"]
        assert hosts == ["gpt-6-sol"]


# ── Host model discovery ────────────────────────────────────────────────────


class TestHostModelDiscovery:
    def test_fetch_raw_models_is_raw_priority_ordered(self, monkeypatch):
        import httpx

        seen = {}

        def _get(url, headers=None, timeout=None):
            seen["url"] = url
            seen["headers"] = headers
            return _FakeModelsResponse(payload={"models": [
                {"slug": "gpt-5.3-codex", "priority": 5},
                {"slug": "gpt-6-sol", "priority": 1},
                {"slug": "gpt-hidden", "priority": 0, "visibility": "hide"},
                {"slug": "  ", "priority": 0},
                "junk",
            ]})

        monkeypatch.setattr(httpx, "get", _get)

        slugs = codex_plugin._fetch_raw_codex_models("codex-token")

        # No synthetic forward-compat entries (gpt-5.5 / gpt-5.4 / spark).
        assert slugs == ["gpt-6-sol", "gpt-5.3-codex"]
        assert seen["url"].startswith("https://chatgpt.com/backend-api/codex/models")
        assert seen["headers"]["Authorization"] == "Bearer codex-token"
        assert seen["headers"]["originator"] == "codex_cli_rs"

    def test_fetch_raw_models_http_error_does_not_leak_token(self, monkeypatch):
        import httpx

        monkeypatch.setattr(
            httpx, "get", lambda *a, **kw: _FakeModelsResponse(status_code=401, payload={})
        )

        with pytest.raises(codex_plugin.CodexHostModelError) as excinfo:
            codex_plugin._fetch_raw_codex_models("secret-codex-token")
        assert "401" in str(excinfo.value)
        assert "secret-codex-token" not in str(excinfo.value)

    def test_fetch_raw_models_network_error_raises(self, monkeypatch):
        import httpx

        def _get(*a, **kw):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr(httpx, "get", _get)

        with pytest.raises(codex_plugin.CodexHostModelError):
            codex_plugin._fetch_raw_codex_models("codex-token")

    def test_fetch_raw_models_network_error_does_not_leak_token(self, monkeypatch, provider):
        import httpx

        token = "secret-codex-token"

        def _get(url, headers=None, timeout=None):
            raise httpx.ConnectError(f"boom with headers {headers!r}")

        monkeypatch.setattr(httpx, "get", _get)
        monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: token)

        with pytest.raises(codex_plugin.CodexHostModelError) as excinfo:
            codex_plugin._fetch_raw_codex_models(token)
        assert "ConnectError" in str(excinfo.value)
        assert token not in str(excinfo.value)
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__suppress_context__ is True

        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "host_model_unavailable"
        assert token not in str(result)

    @pytest.mark.parametrize(
        "bad_priority",
        [float("nan"), float("inf"), float("-inf"), True, "1", None, [1]],
    )
    def test_fetch_raw_models_bad_priority_sorts_last(self, monkeypatch, bad_priority):
        import httpx

        monkeypatch.setattr(
            httpx,
            "get",
            lambda *a, **kw: _FakeModelsResponse(payload={"models": [
                {"slug": "gpt-bad", "priority": bad_priority},
                {"slug": "gpt-6-sol", "priority": 9_999},
            ]}),
        )

        assert codex_plugin._fetch_raw_codex_models("codex-token") == ["gpt-6-sol", "gpt-bad"]

    def test_fetch_raw_models_accepts_json_nan_infinity_priority(self, monkeypatch):
        import json

        import httpx

        body = (
            '{"models": [{"slug": "gpt-nan", "priority": NaN},'
            ' {"slug": "gpt-inf", "priority": Infinity},'
            ' {"slug": "gpt-6-sol", "priority": 2.7}]}'
        )
        monkeypatch.setattr(
            httpx, "get", lambda *a, **kw: _FakeModelsResponse(payload=json.loads(body))
        )

        assert codex_plugin._fetch_raw_codex_models("codex-token") == [
            "gpt-6-sol",
            "gpt-inf",
            "gpt-nan",
        ]

    def test_fetch_raw_models_malformed_body_raises(self, monkeypatch):
        import httpx

        monkeypatch.setattr(
            httpx, "get", lambda *a, **kw: _FakeModelsResponse(payload=ValueError("bad json"))
        )

        with pytest.raises(codex_plugin.CodexHostModelError):
            codex_plugin._fetch_raw_codex_models("codex-token")

    def test_rank_keeps_mainline_in_catalog_order(self):
        assert codex_plugin._rank_host_models(_PLUS_CATALOG) == _PLUS_CATALOG

    def test_rank_drops_non_mainline_variants(self):
        ranked = codex_plugin._rank_host_models([
            "gpt-5.3-codex",
            "gpt-5.4-mini",
            "gpt-5.3-codex-spark",
            "gpt-5.5-pro",
            "o3",
            "gpt-6-sol",
            "gpt-5.5",
        ])
        assert ranked == ["gpt-6-sol", "gpt-5.5"]

    def test_resolve_prefers_first_mainline(self, plus_catalog):
        hosts = codex_plugin._resolve_host_models("codex-token")
        assert hosts[0] == "gpt-6.1-sol"
        assert "gpt-5.4" not in hosts

    def test_resolve_override_present_in_raw_list(self, monkeypatch, plus_catalog):
        _set_host_override(monkeypatch, "gpt-5.6-terra")
        assert codex_plugin._resolve_host_models("codex-token") == ["gpt-5.6-terra"]

    def test_resolve_override_absent_from_raw_list_fails(self, monkeypatch, plus_catalog):
        _set_host_override(monkeypatch, "gpt-5.4")
        with pytest.raises(codex_plugin.CodexHostModelError) as excinfo:
            codex_plugin._resolve_host_models("codex-token")
        assert "gpt-5.4" in str(excinfo.value)

    def test_resolve_empty_catalog_fails(self, monkeypatch):
        monkeypatch.setattr(
            codex_plugin, "_fetch_raw_codex_models", lambda token: [], raising=False
        )
        with pytest.raises(codex_plugin.CodexHostModelError):
            codex_plugin._resolve_host_models("codex-token")

    def test_resolve_catalog_without_mainline_fails(self, monkeypatch):
        monkeypatch.setattr(
            codex_plugin,
            "_fetch_raw_codex_models",
            lambda token: ["gpt-5.3-codex", "gpt-5.4-mini"],
            raising=False,
        )
        with pytest.raises(codex_plugin.CodexHostModelError) as excinfo:
            codex_plugin._resolve_host_models("codex-token")
        assert "gpt-5.3-codex" in str(excinfo.value)

    def test_resolve_does_not_use_synthetic_catalog_helper(self, monkeypatch, plus_catalog):
        import hermes_cli.codex_models as codex_models

        def _forbidden(*a, **kw):
            raise AssertionError("synthetic forward-compat helper must not be used")

        monkeypatch.setattr(codex_models, "get_codex_model_ids", _forbidden)
        monkeypatch.setattr(codex_models, "_fetch_models_from_api", _forbidden)
        assert codex_plugin._resolve_host_models("codex-token")[0] == "gpt-6.1-sol"

    def test_resolve_caches_successful_catalog(self, plus_catalog):
        codex_plugin._resolve_host_models("codex-token")
        codex_plugin._resolve_host_models("codex-token")
        assert plus_catalog == ["codex-token"]
        # Cache keys never hold the raw token.
        assert all("codex-token" not in str(k) for k in codex_plugin._HOST_MODEL_CACHE)

    def test_resolve_cache_is_per_token(self, plus_catalog):
        codex_plugin._resolve_host_models("token-a")
        codex_plugin._resolve_host_models("token-b")
        assert plus_catalog == ["token-a", "token-b"]

    def test_resolve_does_not_cache_failures(self, monkeypatch):
        calls = []

        def _fetch(token):
            calls.append(token)
            if len(calls) == 1:
                raise codex_plugin.CodexHostModelError("temporary")
            return list(_PLUS_CATALOG)

        monkeypatch.setattr(codex_plugin, "_fetch_raw_codex_models", _fetch, raising=False)
        with pytest.raises(codex_plugin.CodexHostModelError):
            codex_plugin._resolve_host_models("codex-token")
        assert codex_plugin._resolve_host_models("codex-token")[0] == "gpt-6.1-sol"
        assert len(calls) == 2

    def test_collect_maps_unsupported_model_400_to_rejection(self, monkeypatch):
        import httpx

        def _handler(request):
            return httpx.Response(
                400,
                json={"detail": "The 'gpt-6.1-sol' model is not supported when using "
                                "Codex with a ChatGPT account."},
            )

        real_client = httpx.Client

        def _client(*a, **kw):
            kw["transport"] = httpx.MockTransport(_handler)
            return real_client(*a, **kw)

        monkeypatch.setattr(httpx, "Client", _client)

        with pytest.raises(codex_plugin.CodexHostModelRejected) as excinfo:
            codex_plugin._collect_image_b64(
                "codex-token", prompt="a cat", size="1024x1024",
                quality="medium", host_model="gpt-6.1-sol",
            )
        assert excinfo.value.host_model == "gpt-6.1-sol"
        assert "codex-token" not in str(excinfo.value)

    def test_collect_other_400_is_plain_error(self, monkeypatch):
        import httpx

        def _handler(request):
            return httpx.Response(400, json={"detail": "prompt rejected by safety system"})

        real_client = httpx.Client

        def _client(*a, **kw):
            kw["transport"] = httpx.MockTransport(_handler)
            return real_client(*a, **kw)

        monkeypatch.setattr(httpx, "Client", _client)

        with pytest.raises(RuntimeError) as excinfo:
            codex_plugin._collect_image_b64(
                "codex-token", prompt="a cat", size="1024x1024",
                quality="medium", host_model="gpt-6-sol",
            )
        assert not isinstance(excinfo.value, codex_plugin.CodexHostModelRejected)
        assert "HTTP 400" in str(excinfo.value)


# ── Plugin entry point ──────────────────────────────────────────────────────


class TestRegistration:
    def test_register_calls_register_image_gen_provider(self):
        registered = []

        class _Ctx:
            def register_image_gen_provider(self, prov):
                registered.append(prov)

        codex_plugin.register(_Ctx())
        assert len(registered) == 1
        assert registered[0].name == "openai-codex"
