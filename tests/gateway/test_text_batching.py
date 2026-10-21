"""Tests for text message batching across all gateway adapters.

When a user sends a long message, the messaging client splits it at the
platform's character limit.  Each adapter should buffer rapid successive
text messages from the same session and aggregate them before dispatching.

Covers: Discord, Matrix, WeCom, and the adaptive delay logic for
Telegram and Feishu.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


# =====================================================================
# Helpers
# =====================================================================

def _make_event(
    text: str,
    platform: Platform,
    chat_id: str = "12345",
    msg_type: MessageType = MessageType.TEXT,
    parent_chat_id: str | None = None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type="dm",
            parent_chat_id=parent_chat_id,
        ),
    )


# =====================================================================
# Discord text batching
# =====================================================================

def _make_discord_adapter(
    channel_text_batch_delays: dict | None = None,
    channel_text_batch_split_delays: dict | None = None,
):
    """Create a minimal DiscordAdapter for testing text batching."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(DiscordAdapter)
    adapter._platform = adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1  # fast for tests
    adapter._text_batch_split_delay_seconds = 0.3  # fast for tests
    adapter._channel_text_batch_delays = channel_text_batch_delays or {}
    adapter._channel_text_batch_split_delays = channel_text_batch_split_delays or {}
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestDiscordTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_discord_adapter()
        event = _make_event("hello world", Platform.DISCORD)

        adapter._enqueue_text_event(event)

        # Not dispatched yet
        adapter.handle_message.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        """Two rapid messages from the same chat should be merged."""
        adapter = _make_discord_adapter()

        adapter._enqueue_text_event(_make_event("Part one of a long", Platform.DISCORD))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("message that was split.", Platform.DISCORD))

        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "Part one" in text
        assert "split" in text


# =====================================================================
# Discord per-channel text batch delays
# =====================================================================

class TestDiscordPerChannelDelay:
    @pytest.mark.asyncio
    async def test_exact_channel_id_uses_per_channel_delay(self):
        """A channel with a configured delay should use that delay instead of the global."""
        adapter = _make_discord_adapter(
            channel_text_batch_delays={"999": 0.5},
        )
        event = _make_event("hello", Platform.DISCORD, chat_id="999")

        adapter._enqueue_text_event(event)

        # Global delay is 0.1s, per-channel is 0.5s — should NOT flush at 0.2s
        await asyncio.sleep(0.2)
        adapter.handle_message.assert_not_called()

        # Should flush after 0.5s (total sleep 0.6s from enqueue)
        await asyncio.sleep(0.4)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_unconfigured_channel_uses_global_delay(self):
        """Channels not listed in per-channel config fall back to the global delay."""
        adapter = _make_discord_adapter(
            channel_text_batch_delays={"999": 0.5},
        )
        event = _make_event("hello", Platform.DISCORD, chat_id="111")

        adapter._enqueue_text_event(event)

        # Global delay 0.1s — should flush after ~0.1s
        await asyncio.sleep(0.2)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_parent_channel_id_fallback(self):
        """Threads whose parent channel has a configured delay should inherit it."""
        adapter = _make_discord_adapter(
            channel_text_batch_delays={"parent_ch": 0.5},
        )
        # chat_id is the thread, parent_chat_id is the parent channel
        event = _make_event(
            "from thread",
            Platform.DISCORD,
            chat_id="thread_ch",
            parent_chat_id="parent_ch",
        )

        adapter._enqueue_text_event(event)

        # Should NOT flush at 0.2s (parent channel delay is 0.5s)
        await asyncio.sleep(0.2)
        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.4)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_exact_match_preferred_over_parent(self):
        """Exact channel match takes precedence over parent channel match."""
        adapter = _make_discord_adapter(
            channel_text_batch_delays={"thread_ch": 0.1, "parent_ch": 0.5},
        )
        event = _make_event(
            "from thread",
            Platform.DISCORD,
            chat_id="thread_ch",
            parent_chat_id="parent_ch",
        )

        adapter._enqueue_text_event(event)

        # Exact match 0.1s — should flush quickly
        await asyncio.sleep(0.2)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_per_channel_split_delay(self):
        """A near-limit chunk should use the per-channel split delay when configured."""
        adapter = _make_discord_adapter(
            channel_text_batch_split_delays={"999": 0.6},
        )
        long_text = "x" * 1950  # >= _SPLIT_THRESHOLD (1900)
        event = _make_event(long_text, Platform.DISCORD, chat_id="999")

        adapter._enqueue_text_event(event)

        # Global split delay is 0.3s, per-channel is 0.6s — not flushed at 0.4s
        await asyncio.sleep(0.4)
        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.3)
        adapter.handle_message.assert_called_once()


# =====================================================================
# Discord per-channel delay validation
# =====================================================================

class TestDiscordChannelDelayValidation:
    """Malformed per-channel delays must be rejected, not silently trusted."""

    @pytest.mark.parametrize("value", [
        True,                 # bool is an int subclass — must not become 1.0
        False,
        -1,
        -0.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        "nan",
        "inf",
        "-Infinity",
        300.5,                # just above the cap
        86400,
        1e12,
        "not-a-number",
        "",
        None,
        [0.5],
        {"delay": 0.5},
    ])
    def test_invalid_values_rejected(self, value):
        from plugins.platforms.discord.adapter import DiscordAdapter

        assert DiscordAdapter._coerce_channel_batch_delay(value) is None

    @pytest.mark.parametrize("value,expected", [
        (0, 0.0),
        (0.0, 0.0),
        (1, 1.0),
        (0.75, 0.75),
        (300, 300.0),
        ("2.5", 2.5),
        (" 2.5 ", 2.5),
    ])
    def test_valid_values_accepted(self, value, expected):
        from plugins.platforms.discord.adapter import DiscordAdapter

        assert DiscordAdapter._coerce_channel_batch_delay(value) == expected

    def test_init_drops_invalid_overrides_and_warns(self, caplog):
        """The real __init__ path must filter overrides, not just the helper."""
        from plugins.platforms.discord.adapter import DiscordAdapter

        config = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "channel_text_batch_delays": {
                    "good": 1.5,
                    "boolish": True,
                    "negative": -2,
                    "nan": float("nan"),
                    "inf": float("inf"),
                    "huge": 999999,
                },
                "channel_text_batch_split_delays": {
                    "good_split": 3.0,
                    "neg_inf": float("-inf"),
                    "junk": "soon",
                },
            },
        )

        with caplog.at_level("WARNING"):
            adapter = DiscordAdapter(config)

        assert adapter._channel_text_batch_delays == {"good": 1.5}
        assert adapter._channel_text_batch_split_delays == {"good_split": 3.0}
        # One warning per rejected entry
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and "text batch delay" in r.getMessage()]
        assert len(warnings) == 7

    @pytest.mark.asyncio
    async def test_invalid_override_falls_back_to_global_delay(self, monkeypatch):
        """A rejected override must not stall (or skip) the flush — global wins."""
        from plugins.platforms.discord.adapter import DiscordAdapter

        monkeypatch.setenv("HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS", "0.1")
        config = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"channel_text_batch_delays": {"999": float("inf")}},
        )
        adapter = DiscordAdapter(config)
        adapter.handle_message = AsyncMock()

        assert adapter._channel_text_batch_delays == {}

        adapter._enqueue_text_event(_make_event("hi", Platform.DISCORD, chat_id="999"))
        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.25)
        adapter.handle_message.assert_called_once()


# =====================================================================
# Matrix text batching
# =====================================================================

def _make_matrix_adapter():
    """Create a minimal MatrixAdapter for testing text batching."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(MatrixAdapter)
    adapter._platform = adapter.platform = Platform.MATRIX
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestMatrixTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_matrix_adapter()
        event = _make_event("hello world", Platform.MATRIX)

        adapter._enqueue_text_event(event)

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        adapter = _make_matrix_adapter()

        adapter._enqueue_text_event(_make_event("first part", Platform.MATRIX))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("second part", Platform.MATRIX))

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "first part" in text
        assert "second part" in text


# =====================================================================
# WeCom text batching
# =====================================================================

def _make_wecom_adapter():
    """Create a minimal WeComAdapter for testing text batching."""
    from plugins.platforms.wecom.adapter import WeComAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(WeComAdapter)
    adapter._platform = adapter.platform = Platform.WECOM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestWeComTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_wecom_adapter()
        event = _make_event("hello world", Platform.WECOM)

        adapter._enqueue_text_event(event)

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        adapter = _make_wecom_adapter()

        adapter._enqueue_text_event(_make_event("first part", Platform.WECOM))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("second part", Platform.WECOM))

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "first part" in text
        assert "second part" in text
