"""Shared fakes and fixtures for the unit-test tier.

Everything here builds on ``mock_context.mock_ctx`` and the stubs that
already exist across the suite (``_Bot``, ``_chunk``, ``_fake_client``,
``setUpModule`` save-stubs), collected in one place so new tests stop
copying them. Import as a top-level module (``from helpers import ...``)
to match the rest of the suite.
"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from settings_store import SettingsStore


class RecordingBot:
    """A bot stub that captures sends, deletions and raw API posts.

    ``sent`` holds the text of every ``send_message``; ``posts`` holds
    ``(method, data)`` pairs from ``_post`` (the media/document path).
    """

    def __init__(self) -> None:
        self.sent: list = []
        self.deleted: list = []
        self.posts: list = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append(text)
        return SimpleNamespace(message_id=7)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))
        return True

    async def send_chat_action(self, *args, **kwargs):
        return True

    async def _post(self, method, data):
        self.posts.append((method, data))
        return {'message_id': 9}


def recording_bot() -> RecordingBot:
    """A fresh RecordingBot (function form, for readability in tests)."""
    return RecordingBot()


def llm_chunk(content=None, reasoning=None, tool_calls=None):
    """One streaming chunk in the shape the OpenAI SDK yields."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=content, reasoning_content=reasoning,
        tool_calls=tool_calls), finish_reason=None)])


def fake_client(captured: dict, chunks: list):
    """A client whose ``chat.completions.create`` streams ``chunks``.

    ``captured`` receives the kwargs of the ``create`` call, so tests
    can assert on the exact request payload.
    """
    async def fake_create(**kwargs):
        captured.update(kwargs)

        async def stream():
            for c in chunks:
                yield c
        return stream()
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=fake_create)))


def temp_settings_db(test: unittest.TestCase) -> SettingsStore:
    """A fresh settings store over a throwaway file, closed at cleanup.

    Use for tests that need a real SettingsStore but not the process's
    isolated test database (which the tier already provides via
    ``config.sql_db_path``).
    """
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    async def build() -> SettingsStore:
        store = SettingsStore(path)
        await store.create_schema()
        await store.close()
        return store

    import asyncio
    store = asyncio.run(build())
    test.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
    return store


def stub_save_conversation(test: unittest.TestCase, module):
    """Stub a module's ``save_conversation`` with an AsyncMock.

    The generation worker persists every queued job; persistence itself
    is covered by test_conversation_persistence.py, so handler tests
    stub it to stay hermetic and fast.
    """
    patcher = mock.patch.object(module, 'save_conversation',
                                mock.AsyncMock())
    patcher.start()
    test.addCleanup(patcher.stop)
