import base64
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import requests
from conversation import Conversation
from messaging import Messaging
from tools import image_gen
from tool_result import ToolCallResult
from mock_context import mock_ctx


class TestImageGenErrors(unittest.IsolatedAsyncioTestCase):
    # The model only sees the tool result string, so a generation failure has
    # to carry the endpoint's own error text back to it.

    async def _run(self, error: Exception) -> str:
        async def boom(*args, **kwargs):
            raise error
        with unittest.mock.patch.object(image_gen, 'generate_image', boom):
            result = await image_gen.generate_and_send_image(mock_ctx(), 'a cat')
        assert isinstance(result, str)
        return result

    async def test_http_error_body_is_returned(self):
        response = requests.Response()
        response.status_code = 400
        response.encoding = 'utf-8'
        response._content = b'{"detail":"content policy violated"}'
        result = await self._run(
            requests.HTTPError('400 Client Error', response=response))
        self.assertEqual(
            result, 'Failed to generate image: {"detail":"content policy violated"}')

    async def test_generic_error_message_is_returned(self):
        result = await self._run(RuntimeError('Generation blocked by content filter'))
        self.assertEqual(
            result, 'Failed to generate image: Generation blocked by content filter')


class TestParseResolution(unittest.IsolatedAsyncioTestCase):
    # Pure logic: WIDTHxHEIGHT validation, valid both directions.

    def test_valid_dimensions(self):
        self.assertEqual(image_gen.parse_resolution('1024x1024'),
                         (1024, 1024))
        self.assertEqual(image_gen.parse_resolution(' 1568 X 672 '),
                         (1568, 672))
        self.assertEqual(image_gen.parse_resolution('768*1344'), (768, 1344))

    def test_invalid_dimensions(self):
        for bad in ('nonsense', '1024', '512x100', '1025x1024', '0x512'):
            with self.assertRaises(ValueError):
                image_gen.parse_resolution(bad)

    async def test_invalid_resolution_reported_to_the_model(self):
        result = await image_gen.generate_and_send_image(
            mock_ctx(), 'a cat', resolution='big')
        assert isinstance(result, str)
        self.assertIn("Invalid resolution 'big'", result)
        self.assertNotIn('a cat', result)  # generation never starts


class TestSendToChat(unittest.IsolatedAsyncioTestCase):
    # send_to_chat (default true) controls whether the user sees the image;
    # either way the image is stored in the sandbox for further processing.

    STORED_PATH = '/workspace/uploads/generated_1.png'

    async def _run(self, store_error=None, **kwargs):
        async def fake_generate(*args, **_):
            return base64.b64encode(b'\x89PNG').decode()

        async def fake_store(user_id, data, name):
            if store_error is not None:
                return None, store_error
            return self.STORED_PATH, None

        send = unittest.mock.AsyncMock()
        with unittest.mock.patch.object(image_gen, 'generate_image', fake_generate):
            with unittest.mock.patch.object(Messaging, 'send_image', send):
                with unittest.mock.patch.object(
                        image_gen.manager, 'store_file', fake_store):
                    result = await image_gen.generate_and_send_image(
                        mock_ctx(), 'a cat', **kwargs)
        assert isinstance(result, str)
        return result, send

    async def test_send_to_chat_branches(self):
        # By default the image is sent to the chat...
        result, send = await self._run()
        send.assert_awaited_once()
        call = send.await_args
        assert call is not None
        self.assertEqual(call.args[0], b'\x89PNG')
        self.assertIn('sent to chat and stored', result)
        self.assertIn(self.STORED_PATH, result)

        # ...and send_to_chat=False keeps it private instead. Either way
        # the image is stored in the sandbox for further processing.
        result, send = await self._run(send_to_chat=False)
        send.assert_not_awaited()
        self.assertIn(f'stored in the sandbox at {self.STORED_PATH}', result)
        self.assertNotIn('sent to chat', result)

    async def test_store_error_reports_delivery(self):
        result, send = await self._run(store_error='disk full')
        self.assertEqual(result, 'Image sent to chat (prompt: a cat)')

    async def test_store_error_without_sending(self):
        result, send = await self._run(store_error='disk full', send_to_chat=False)
        self.assertEqual(result, 'Image generated (prompt: a cat)')


class TestAttachToContext(unittest.IsolatedAsyncioTestCase):
    # add_to_context (default false) attaches the generated image to the
    # conversation so a vision-capable model can inspect it; otherwise the
    # tool result stays a plain string.

    STORED_PATH = '/workspace/uploads/generated_1.png'

    async def _run(self, **kwargs):
        async def fake_generate(*args, **_):
            return base64.b64encode(b'\x89PNG').decode()

        async def fake_store(user_id, data, name):
            return self.STORED_PATH, None

        with unittest.mock.patch.object(image_gen, 'generate_image', fake_generate):
            with unittest.mock.patch.object(Messaging, 'send_image',
                                            unittest.mock.AsyncMock()):
                with unittest.mock.patch.object(
                        image_gen.manager, 'store_file', fake_store):
                    return await image_gen.generate_and_send_image(
                        mock_ctx(), 'a cat', **kwargs)

    async def test_default_is_plain_string(self):
        result = await self._run()
        self.assertIsInstance(result, str)

    async def test_attach_returns_image_result(self):
        result = await self._run(add_to_context=True)
        assert isinstance(result, ToolCallResult)
        assert result.image is not None
        self.assertEqual(result.image.data, b'\x89PNG')
        self.assertEqual(result.image.mime, 'image/png')
        self.assertIn(self.STORED_PATH, result.text)


class TestSubagentVariant(unittest.IsolatedAsyncioTestCase):
    # The subagent variant shares the business logic but has no
    # send_to_chat parameter and never posts: a silent context has no
    # messaging handle at all.

    STORED_PATH = '/workspace/uploads/generated_1.png'

    async def _run(self):
        async def fake_generate(*args, **_):
            return base64.b64encode(b'\x89PNG').decode()

        async def fake_store(user_id, data, name):
            return self.STORED_PATH, None

        send = unittest.mock.AsyncMock()
        ctx = mock_ctx().new_subagent_context(Conversation())
        with unittest.mock.patch.object(image_gen, 'generate_image', fake_generate):
            with unittest.mock.patch.object(Messaging, 'send_image', send):
                with unittest.mock.patch.object(
                        image_gen.manager, 'store_file', fake_store):
                    result = await image_gen.generate_image_subagent(
                        ctx, 'a cat')
        return result, send

    def test_schema_has_no_send_switch(self):
        from tool_registry import SUBAGENT, tools_for
        subagent = next(t for t in tools_for(SUBAGENT)
                        if t.name == 'generate_image')
        self.assertNotIn('send_to_chat', subagent.properties)
        self.assertIn('add_to_context', subagent.properties)

    def test_both_roles_see_the_same_name(self):
        from tool_registry import PARENT, SUBAGENT, tools_for
        parent = next(t for t in tools_for(PARENT)
                      if t.name == 'generate_image')
        subagent = next(t for t in tools_for(SUBAGENT)
                        if t.name == 'generate_image')
        self.assertIn('send_to_chat', parent.properties)
        self.assertIsNot(parent.func, subagent.func)

    async def test_never_posts_and_reports_the_path(self):
        result, send = await self._run()
        send.assert_not_awaited()
        assert isinstance(result, str)
        self.assertIn(f'stored in the sandbox at {self.STORED_PATH}', result)
        self.assertNotIn('sent to chat', result)


if __name__ == '__main__':
    unittest.main()
