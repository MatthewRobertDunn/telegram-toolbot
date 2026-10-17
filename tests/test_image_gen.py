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


if __name__ == '__main__':
    unittest.main()
