import base64
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import requests
from tools import image_gen
from mock_context import mock_ctx


class TestImageGenErrors(unittest.IsolatedAsyncioTestCase):
    # The model only sees the tool result string, so a generation failure has
    # to carry the endpoint's own error text back to it.

    async def _run(self, error: Exception) -> str:
        async def boom(*args, **kwargs):
            raise error
        with unittest.mock.patch.object(image_gen, 'generate_image', boom):
            return await image_gen.generate_and_send_image(mock_ctx(), 'a cat')

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


class TestSendToChat(unittest.IsolatedAsyncioTestCase):
    # send_to_chat (default true) controls whether the user sees the image;
    # either way the image is stored in the sandbox for further processing.

    STORED_PATH = '/workspace/uploads/generated_1.png'

    async def _run(self, store_error=None, **kwargs):
        sent: list = []

        async def fake_generate(*args, **_):
            return base64.b64encode(b'\x89PNG').decode()

        async def fake_send_image(ctx, data):
            sent.append(data)

        async def fake_store(user_id, data, name):
            if store_error is not None:
                return None, store_error
            return self.STORED_PATH, None

        with unittest.mock.patch.object(image_gen, 'generate_image', fake_generate):
            with unittest.mock.patch.object(image_gen, 'send_image', fake_send_image):
                with unittest.mock.patch.object(
                        image_gen.manager, 'store_file', fake_store):
                    result = await image_gen.generate_and_send_image(
                        mock_ctx(), 'a cat', **kwargs)
        return result, sent

    async def test_defaults_to_sending(self):
        result, sent = await self._run()
        self.assertEqual(len(sent), 1)
        self.assertIn('sent to chat and stored', result)
        self.assertIn(self.STORED_PATH, result)

    async def test_send_to_chat_false_stores_without_sending(self):
        result, sent = await self._run(send_to_chat=False)
        self.assertEqual(sent, [])
        self.assertIn(f'stored in the sandbox at {self.STORED_PATH}', result)
        self.assertNotIn('sent to chat', result)

    async def test_store_error_reports_delivery(self):
        result, sent = await self._run(store_error='disk full')
        self.assertEqual(result, 'Image sent to chat (prompt: a cat)')

    async def test_store_error_without_sending(self):
        result, sent = await self._run(store_error='disk full', send_to_chat=False)
        self.assertEqual(result, 'Image generated (prompt: a cat)')


if __name__ == '__main__':
    unittest.main()
