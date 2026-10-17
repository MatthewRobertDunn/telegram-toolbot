import time
import unittest
import unittest.mock
from types import SimpleNamespace

import bootstrap  # noqa: F401 — path setup, env, logging, DB reset
from bootstrap import next_uid, requires_docker
from mocks import RecordingBot

from chat_update_context import ChatUpdateContext
from handlers.upload_handler import upload_handler
from sandbox import manager


def _upload_ctx(user_id: int, media, caption=None) -> ChatUpdateContext:
    bot = RecordingBot()
    context = SimpleNamespace(
        bot=bot, bot_data={"allowed_ids": []}, user_data={}, chat_data={})
    message = SimpleNamespace(
        text=None, caption=caption,
        chat=SimpleNamespace(id=user_id, type="private"),
        from_user=SimpleNamespace(id=user_id, first_name="T", last_name="U"),
        document=media,
        photo=None, audio=None, video=None, voice=None,
        video_note=None, animation=None, sticker=None)
    return ChatUpdateContext(
        update=None,  # type: ignore[arg-type]
        context=context,  # type: ignore[arg-type]
        bot=bot,  # type: ignore[arg-type]
        chat=message.chat,
        user=message.from_user,
        message=message,  # type: ignore[arg-type]
        user_data=context.user_data,
        chat_data=None,
        command_text='',
        command_name='')


def _document(file_name=None):
    class Document:
        file_id: str
        file_name: str | None
    media = Document()
    media.file_id = "fid123"
    media.file_name = file_name
    return media


async def _fake_run_llm(ctx, **kwargs):
    return "Upload received."


@requires_docker
class TestUploadHandler(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.uid = next_uid()
        await manager.destroy(self.uid)

    async def asyncTearDown(self):
        await manager.destroy(self.uid)

    async def test_document_upload_stored_and_llm_notified(self):
        ctx = _upload_ctx(self.uid, _document("hello.txt"))
        with unittest.mock.patch("handlers.upload_handler.run_llm",
                                 _fake_run_llm):
            await upload_handler(ctx)

        result = await manager.exec(
            self.uid, "cat /workspace/uploads/hello.txt")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "mock file contents")

        user_messages = [m for m in ctx.conversation if m["role"] == "user"]
        self.assertTrue(any("/workspace/uploads/hello.txt" in m["content"]
                            and "read tool" in m["content"]
                            for m in user_messages))
        self.assertEqual(
            ctx.bot.sent[-1]["text"], "Upload received.")  # type: ignore[attr-defined]

        # The upload starts the sandbox and counts as activity.
        running = await manager._container_running(manager._sandboxes[self.uid])
        self.assertTrue(running)
        self.assertGreater(
            manager._sandboxes[self.uid].last_activity, time.time() - 60)

    async def test_upload_without_name_gets_fallback(self):
        ctx = _upload_ctx(self.uid, _document(None))
        with unittest.mock.patch("handlers.upload_handler.run_llm",
                                 _fake_run_llm):
            await upload_handler(ctx)
        listing = await manager.exec(
            self.uid, "ls -1 /workspace/uploads")
        files = listing["stdout"].split()
        self.assertEqual(len(files), 1)
        self.assertRegex(files[0], r"^document_\d+$")

    async def test_name_collision_gets_suffix(self):
        for _ in range(2):
            ctx = _upload_ctx(self.uid, _document("hello.txt"))
            with unittest.mock.patch("handlers.upload_handler.run_llm",
                                     _fake_run_llm):
                await upload_handler(ctx)
        listing = await manager.exec(
            self.uid, "ls -1 /workspace/uploads")
        self.assertEqual(
            sorted(listing["stdout"].split()), ["hello-1.txt", "hello.txt"])

    async def test_caption_included_in_notification(self):
        ctx = _upload_ctx(self.uid, _document("notes.txt"), caption="please review")
        with unittest.mock.patch("handlers.upload_handler.run_llm",
                                 _fake_run_llm):
            await upload_handler(ctx)
        user_messages = [m for m in ctx.conversation if m["role"] == "user"]
        last_user = user_messages[-1]["content"]
        self.assertTrue(last_user.startswith("please review"))
        self.assertIn("/workspace/uploads/notes.txt", last_user)


if __name__ == "__main__":
    unittest.main()
