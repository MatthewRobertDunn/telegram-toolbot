import base64
import json
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


class TestProviderParam(unittest.IsolatedAsyncioTestCase):
    # The provider parameter selects the image backend; the default is
    # Cloudflare Phoenix.

    def test_schema_exposes_provider_for_both_roles(self):
        from tool_registry import PARENT, SUBAGENT, tools_for
        for role in (PARENT, SUBAGENT):
            tool = next(t for t in tools_for(role)
                        if t.name == 'generate_image')
            self.assertIn('provider', tool.properties)
            self.assertEqual(tool.properties['provider']['enum'],
                             ['nvidia-flux3', 'nvidia-flux2', 'horde',
                              'cloudflare'])

    def test_default_provider_is_a_known_provider(self):
        self.assertIn(image_gen.DEFAULT_PROVIDER, image_gen.IMAGE_PROVIDERS)

    async def test_unknown_provider_reported_to_the_model(self):
        result = await image_gen.generate_and_send_image(
            mock_ctx(), 'a cat', provider='bogus')
        assert isinstance(result, str)
        self.assertIn("Unknown image provider 'bogus'", result)

    async def test_provider_reaches_the_backend(self):
        seen = {}

        async def fake_generate(prompt, width=1024, height=1024, seed=0,
                                steps=None, provider='unset',
                                guidance=None, negative_prompt=None):
            seen['provider'] = provider
            return base64.b64encode(b'\x89PNG').decode()

        async def fake_store(user_id, data, name):
            return '/workspace/uploads/generated_1.png', None

        with unittest.mock.patch.object(image_gen, 'generate_image',
                                        fake_generate):
            with unittest.mock.patch.object(Messaging, 'send_image',
                                            unittest.mock.AsyncMock()):
                with unittest.mock.patch.object(image_gen.manager,
                                               'store_file', fake_store):
                    await image_gen.generate_and_send_image(
                        mock_ctx(), 'a cat', provider='horde')
        self.assertEqual(seen.get('provider'), 'horde')


class TestCloudflareProvider(unittest.IsolatedAsyncioTestCase):
    # Phoenix params: prompt/width/height/num_steps, plus optional
    # guidance (2-10), seed and negative_prompt. The endpoint answers
    # with raw JPEG bytes or with {"result": {"image": "<base64>"}}.

    def _response(self, body: bytes, content_type: str):
        response = requests.Response()
        response.status_code = 200
        response.headers['Content-Type'] = content_type
        response._content = body
        response.encoding = 'utf-8'
        return response

    async def _generate(self, response, **kwargs):
        with unittest.mock.patch.object(
                image_gen.config, 'CLOUDFLARE_ACCOUNT_ID', 'acct'), \
             unittest.mock.patch.object(
                image_gen.config, 'CLOUDFLARE_API_TOKEN', 'tok'), \
             unittest.mock.patch.object(
                image_gen.requests, 'post',
                return_value=response) as post:
            result = await image_gen._generate_cloudflare('a cat', **kwargs)
        return result, post

    async def test_raw_jpeg_bytes_are_base64_encoded(self):
        result, post = await self._generate(
            self._response(b'\xff\xd8\xffdata', 'image/jpeg'))
        self.assertEqual(result, base64.b64encode(b'\xff\xd8\xffdata').decode())
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['num_steps'], 25)
        self.assertNotIn('guidance', payload)
        self.assertNotIn('negative_prompt', payload)
        self.assertNotIn('seed', payload)

    async def test_json_base64_shape_and_optional_params(self):
        body = ('{"result": {"image": "%s"}}'
                % base64.b64encode(b'\xff\xd8\xffdata').decode()).encode()
        result, post = await self._generate(
            self._response(body, 'application/json'),
            steps=30, seed=7, guidance=5,
            negative_prompt='blurry')
        self.assertEqual(result, base64.b64encode(b'\xff\xd8\xffdata').decode())
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['num_steps'], 30)
        self.assertEqual(payload['seed'], 7)
        self.assertEqual(payload['guidance'], 5)
        self.assertEqual(payload['negative_prompt'], 'blurry')

    async def test_invalid_steps_and_guidance_rejected(self):
        response = self._response(b'\xff\xd8\xffdata', 'image/jpeg')
        with self.assertRaises(ValueError):
            await self._generate(response, steps=0)
        with self.assertRaises(ValueError):
            await self._generate(response, guidance=11)

    async def test_missing_credentials_reported(self):
        with unittest.mock.patch.object(
                image_gen.config, 'CLOUDFLARE_ACCOUNT_ID', ''), \
             unittest.mock.patch.object(
                image_gen.config, 'CLOUDFLARE_API_TOKEN', 'tok'):
            with self.assertRaises(RuntimeError):
                await image_gen._generate_cloudflare('a cat')

    async def test_dispatch_uses_cloudflare_defaults(self):
        seen = {}

        async def fake_cloudflare(prompt, width=1024, height=1024, seed=0,
                                  steps=25, guidance=None,
                                  negative_prompt=None):
            seen.update(steps=steps, guidance=guidance,
                        negative_prompt=negative_prompt)
            return base64.b64encode(b'\xff\xd8\xffdata').decode()

        with unittest.mock.patch.object(image_gen, '_generate_cloudflare',
                                        fake_cloudflare):
            result = await image_gen.generate_image(
                'a cat', provider='cloudflare')
        self.assertEqual(
            result, base64.b64encode(b'\xff\xd8\xffdata').decode())
        self.assertEqual(
            seen, {'steps': 25, 'guidance': None, 'negative_prompt': None})


class TestNvidiaProvider(unittest.IsolatedAsyncioTestCase):
    # The FLUX-3 NIM endpoint rejects a ``seed`` field (HTTP 422
    # ``extra_forbidden``), so no seed is ever sent.

    async def test_payload_has_no_seed(self):
        body = base64.b64encode(b'\x89PNG\r\n\x1a\napple').decode()
        response = requests.Response()
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json'
        response._content = (
            ('{"artifacts": [%s]}' % json.dumps(
                {'base64': body, 'finishReason': 'SUCCESS'})).encode())
        response.encoding = 'utf-8'
        with unittest.mock.patch.object(
                image_gen.requests, 'post',
                return_value=response) as post:
            result = await image_gen._generate_nvidia_flux3('an apple')
        self.assertEqual(result, body)
        self.assertEqual(post.call_args.args[0],
                         image_gen.config.IMAGE_GEN_URL)
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['steps'], 50)
        self.assertNotIn('seed', payload)


class TestNvidiaFlux2Provider(unittest.IsolatedAsyncioTestCase):
    # FLUX.2-Klein posts to its own NIM URL with prompt/width/height/steps
    # and no seed (the endpoint 422s on it). Response shape matches the
    # FLUX-3 endpoint: {"artifacts": [{"base64": ..., "finishReason": ...}]}.

    def _response(self, artifact: dict):
        response = requests.Response()
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json'
        response._content = (
            ('{"artifacts": [%s]}' % json.dumps(artifact)).encode())
        response.encoding = 'utf-8'
        return response

    async def test_payload_has_no_seed_and_defaults_to_4_steps(self):
        body = base64.b64encode(b'\x89PNG\r\n\x1a\nfrog').decode()
        with unittest.mock.patch.object(
                image_gen.requests, 'post',
                return_value=self._response(
                    {'base64': body, 'finishReason': 'SUCCESS'})) as post:
            result = await image_gen._generate_nvidia_flux2('a frog')
        self.assertEqual(result, body)
        self.assertEqual(post.call_args.args[0],
                         image_gen.config.NVIDIA_FLUX2_GEN_URL)
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['steps'], 4)
        self.assertNotIn('seed', payload)

    async def test_content_filter_raises(self):
        body = base64.b64encode(b'nope').decode()
        with unittest.mock.patch.object(
                image_gen.requests, 'post',
                return_value=self._response(
                    {'base64': body, 'finishReason': 'CONTENT_FILTERED'})):
            with self.assertRaises(RuntimeError):
                await image_gen._generate_nvidia_flux2('a frog')

    async def test_dispatch_defaults_to_4_steps(self):
        seen = {}

        async def fake_flux2(prompt, width=1024, height=1024, steps=4):
            seen['steps'] = steps
            return 'b64'

        with unittest.mock.patch.object(image_gen, '_generate_nvidia_flux2',
                                        fake_flux2):
            result = await image_gen.generate_image(
                'a frog', provider='nvidia-flux2')
        self.assertEqual(result, 'b64')
        self.assertEqual(seen.get('steps'), 4)


class TestSniffFormat(unittest.IsolatedAsyncioTestCase):
    # Backends disagree on containers (NVIDIA PNG, Horde WEBP); the stored
    # filename and context attachment must match the actual bytes.

    def test_known_magics(self):
        self.assertEqual(
            image_gen._sniff_format(b'\x89PNG\r\n\x1a\nrest'),
            ('png', 'image/png'))
        self.assertEqual(image_gen._sniff_format(b'\xff\xd8\xffrest'),
                         ('jpg', 'image/jpeg'))
        self.assertEqual(
            image_gen._sniff_format(b'RIFF\x00\x00\x00\x00WEBP rest'),
            ('webp', 'image/webp'))

    def test_unknown_defaults_to_png(self):
        self.assertEqual(image_gen._sniff_format(b'\x89PNG'),
                         ('png', 'image/png'))

    async def test_webp_stored_with_webp_extension_and_mime(self):
        async def fake_generate(*args, **_):
            return base64.b64encode(b'RIFF\x00\x00\x00\x00WEBPdata').decode()

        names = []

        async def fake_store(user_id, data, name):
            names.append(name)
            return '/workspace/uploads/' + name, None

        with unittest.mock.patch.object(image_gen, 'generate_image',
                                        fake_generate):
            with unittest.mock.patch.object(Messaging, 'send_image',
                                            unittest.mock.AsyncMock()):
                with unittest.mock.patch.object(Messaging, 'send_document',
                                                unittest.mock.AsyncMock()):
                    with unittest.mock.patch.object(image_gen.manager,
                                                   'store_file', fake_store):
                        result = await image_gen.generate_and_send_image(
                            mock_ctx(), 'a cat', add_to_context=True)
        assert isinstance(result, ToolCallResult)
        assert result.image is not None
        self.assertEqual(result.image.mime, 'image/webp')
        self.assertTrue(names[0].endswith('.webp'), names)

    async def test_webp_sent_as_document_not_photo(self):
        async def fake_generate(*args, **_):
            return base64.b64encode(b'RIFF\x00\x00\x00\x00WEBPdata').decode()

        async def fake_store(user_id, data, name):
            return '/workspace/uploads/' + name, None

        photo = unittest.mock.AsyncMock()
        doc = unittest.mock.AsyncMock()
        with unittest.mock.patch.object(image_gen, 'generate_image',
                                        fake_generate):
            with unittest.mock.patch.object(Messaging, 'send_image', photo):
                with unittest.mock.patch.object(Messaging, 'send_document',
                                                doc):
                    with unittest.mock.patch.object(image_gen.manager,
                                                   'store_file', fake_store):
                        await image_gen.generate_and_send_image(
                            mock_ctx(), 'a cat')
        photo.assert_not_awaited()
        doc.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
