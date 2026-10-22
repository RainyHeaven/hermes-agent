"""A batch still inside a long per-channel delay must be dispatched at shutdown, not dropped.

``channel_text_batch_delays`` (fork customization C2) can exceed the shutdown flush deadline. Waiting
the delay out would let the deadline cancel the flush task and lose the user's message.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_discord_pending_text_batch_shutdown import DiscordAdapter


def _event(text: str, chat_id: str) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id=chat_id, chat_type="group"),
    )


@pytest.mark.asyncio
async def test_shutdown_dispatches_a_batch_still_waiting_out_a_long_channel_delay(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._channel_text_batch_delays = {"111": 60.0}
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr(adapter, "_text_batch_flush_deadline_seconds", lambda: 0.2)
    monkeypatch.setattr(adapter, "_text_batch_key", lambda event: event.source.chat_id)

    adapter._enqueue_text_event(_event("typed just before restart", "111"))
    await asyncio.sleep(0)  # let the flush task start its 60 s quiet period

    await adapter.cancel_background_tasks()

    adapter.handle_message.assert_awaited_once()
    (dispatched,), _ = adapter.handle_message.await_args
    assert dispatched.text == "typed just before restart"
    assert adapter._pending_text_batches == {}
    assert adapter._pending_text_batch_tasks == {}


@pytest.mark.asyncio
async def test_shutdown_does_not_dispatch_a_batch_twice(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._channel_text_batch_delays = {"111": 60.0}
    adapter.handle_message = AsyncMock()
    monkeypatch.setattr(adapter, "_text_batch_flush_deadline_seconds", lambda: 0.2)
    monkeypatch.setattr(adapter, "_text_batch_key", lambda event: event.source.chat_id)

    adapter._enqueue_text_event(_event("first", "111"))
    adapter._enqueue_text_event(_event("second", "111"))
    await asyncio.sleep(0)

    await adapter.cancel_background_tasks()

    adapter.handle_message.assert_awaited_once()
    (dispatched,), _ = adapter.handle_message.await_args
    assert "first" in dispatched.text and "second" in dispatched.text
