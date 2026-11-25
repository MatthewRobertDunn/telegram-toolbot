import asyncio
import base64
import contextlib
import json
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import http_client
from conversation import Conversation
from messaging import Messaging
from tools import image_gen
from tool_result import ToolCallResult
from mock_context import mock_ctx

# Config values every provider's credential check looks for; tests patch
# them all so no test depends on the real environment.
_CREDENTIALS = dict(
    NVIDIA_API_KEY='nvidia-key',
    CLOUDFLARE_ACCOUNT_ID='acct',
    CLOUDFLARE_API_TOKEN='tok',
    OPENROUTER_API_KEY='or-key')


def _response(body: bytes, content_type: str = 'application/json',
              status: int = 200,
              url: str = 'https://unit.test') -> http_client.Response:
    return http_client.Response(
        status=status, headers={'Content-Type': content_type},
        content=body, url=url)


@contextlib.contextmanager
def _configured(**overrides):
    """Patch the config settings the providers' credential checks read."""
    values = {**_CREDENTIALS, **overrides}
    with contextlib.ExitStack() as stack:
        for attr, value in values.items():
            stack.enter_context(unittest.mock.patch.object(
                image_gen.config, attr, value))
        yield


class TestImageGenErrors(unittest.IsolatedAsyncioTestCase):
    # The model only sees the tool result string, so a generation failure has
    # to carry the endpoint's own error text back to it, and every failure
    # goes through the one truncating formatter.

    async def _run(self, error: Exception) -> str:
        async def boom(*args, **kwargs):
            raise error
        with unittest.mock.patch.object(image_gen, 'generate_image', boom):
            result = await image_gen.generate_and_send_image(mock_ctx(), 'a cat')
        assert isinstance(result, str)
        return result

    async def test_http_error_body_is_returned(self):
        result = await self._run(http_client.HttpStatusError(
            400, 'https://img.test', '{"detail":"content policy violated"}'))
        self.assertEqual(
            result,
            'Failed to generate image: {"detail":"content policy violated"}')

    async def test_http_error_without_a_body_names_the_status(self):
        result = await self._run(
            http_client.HttpStatusError(502, 'https://img.test', ''))
        self.assertEqual(
            result, 'Failed to generate image: HTTP 502 from https://img.test')

    async def test_generic_error_message_is_returned(self):
        result = await self._run(
            RuntimeError('Generation blocked by content filter'))
        self.assertEqual(
            result, 'Failed to generate image: Generation blocked by content filter')

    async def test_long_error_text_is_truncated(self):
        prefix = 'Failed to generate image: '
        for error in (
                http_client.HttpStatusError(400, 'https://img.test',
                                            'x' * (image_gen.MAX_ERROR_CHARS + 1)),
                RuntimeError('y' * (image_gen.MAX_ERROR_CHARS + 1))):
            result = await self._run(error)
            self.assertEqual(len(result),
                             len(prefix) + image_gen.MAX_ERROR_CHARS)


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
            return b'\x89PNG\r\n\x1a\n'

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
        self.assertEqual(call.args[0], b'\x89PNG\r\n\x1a\n')
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
            return b'\x89PNG\r\n\x1a\n'

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
        self.assertEqual(result.image.data, b'\x89PNG\r\n\x1a\n')
        self.assertEqual(result.image.mime, 'image/png')
        self.assertIn(self.STORED_PATH, result.text)


class TestSubagentVariant(unittest.IsolatedAsyncioTestCase):
    # The subagent variant shares the business logic but has no
    # send_to_chat parameter and never posts: a silent context has no
    # messaging handle at all.

    STORED_PATH = '/workspace/uploads/generated_1.png'

    async def _run(self):
        async def fake_generate(*args, **_):
            return b'\x89PNG\r\n\x1a\n'

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
    # NVIDIA FLUX.3.

    def test_schema_exposes_provider_for_both_roles(self):
        from tool_registry import PARENT, SUBAGENT, tools_for
        for role in (PARENT, SUBAGENT):
            tool = next(t for t in tools_for(role)
                        if t.name == 'generate_image')
            self.assertIn('provider', tool.properties)
            self.assertEqual(tool.properties['provider']['enum'],
                             ['nvidia-flux3', 'nvidia-flux2', 'phoenix',
                              'recraft'])

    def test_default_provider_is_a_known_provider(self):
        self.assertIn(image_gen.DEFAULT_PROVIDER, image_gen.IMAGE_PROVIDERS)

    async def test_unknown_provider_reported_to_the_model(self):
        result = await image_gen.generate_and_send_image(
            mock_ctx(), 'a cat', provider='bogus')
        assert isinstance(result, str)
        self.assertIn("Unknown image provider 'bogus'", result)

    async def test_provider_reaches_the_backend(self):
        seen = {}

        async def fake_generate(prompt, width=1024, height=1024,
                                steps=None, provider='unset'):
            seen['provider'] = provider
            return b'\x89PNG\r\n\x1a\n'

        async def fake_store(user_id, data, name):
            return '/workspace/uploads/generated_1.png', None

        with unittest.mock.patch.object(image_gen, 'generate_image',
                                        fake_generate):
            with unittest.mock.patch.object(Messaging, 'send_image',
                                            unittest.mock.AsyncMock()):
                with unittest.mock.patch.object(image_gen.manager,
                                               'store_file', fake_store):
                    await image_gen.generate_and_send_image(
                        mock_ctx(), 'a cat', provider='phoenix')
        self.assertEqual(seen.get('provider'), 'phoenix')


class TestProviderDispatch(unittest.IsolatedAsyncioTestCase):
    # generate_image resolves each provider's URL, default steps and
    # request shape; the render functions own the wire format.

    def _nvidia_body(self, image: bytes, finish: str = 'SUCCESS') -> bytes:
        return ('{"artifacts": [%s]}' % json.dumps(
            {'base64': base64.b64encode(image).decode(),
             'finishReason': finish})).encode()

    async def _post(self, response: http_client.Response, provider: str,
                    **kwargs) -> tuple[bytes, unittest.mock.AsyncMock]:
        post = unittest.mock.AsyncMock(return_value=response)
        with _configured(), \
             unittest.mock.patch.object(http_client, 'post', post):
            data = await image_gen.generate_image('a cat', provider=provider,
                                                  **kwargs)
        return data, post

    async def test_nvidia_flux3_url_and_default_steps(self):
        # FLUX-3 renders 50 steps by default and never sends a seed (the
        # endpoint 422s an extra_forbidden field).
        image = b'\x89PNG\r\n\x1a\napple'
        data, post = await self._post(
            _response(self._nvidia_body(image)), 'nvidia-flux3')
        self.assertEqual(data, image)
        self.assertEqual(post.call_args.args[0],
                         image_gen.config.IMAGE_GEN_URL)
        self.assertEqual(post.call_args.kwargs['json_body'],
                         {'prompt': 'a cat', 'width': 1024, 'height': 1024,
                          'steps': 50})

    async def test_nvidia_flux2_url_and_default_steps(self):
        # FLUX.2-Klein posts to its own NIM URL with the same payload shape
        # but a 4-step default.
        image = b'\x89PNG\r\n\x1a\nfrog'
        data, post = await self._post(
            _response(self._nvidia_body(image)), 'nvidia-flux2')
        self.assertEqual(data, image)
        self.assertEqual(post.call_args.args[0],
                         image_gen.config.NVIDIA_FLUX2_GEN_URL)
        self.assertEqual(post.call_args.kwargs['json_body'],
                         {'prompt': 'a cat', 'width': 1024, 'height': 1024,
                          'steps': 4})

    async def test_nvidia_explicit_steps_are_sent(self):
        _, post = await self._post(
            _response(self._nvidia_body(b'\x89PNG\r\n\x1a\nx')),
            'nvidia-flux3', steps=10)
        self.assertEqual(post.call_args.kwargs['json_body']['steps'], 10)

    async def test_content_filter_is_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            await self._post(
                _response(self._nvidia_body(b'nope', 'CONTENT_FILTERED')),
                'nvidia-flux2')
        self.assertIn('content filter', str(ctx.exception))

    async def test_phoenix_defaults_to_25_steps(self):
        # Phoenix takes the step count as num_steps; without a JSON content
        # type the raw body bytes are the image.
        image = b'\xff\xd8\xffdata'
        data, post = await self._post(
            _response(image, 'image/jpeg'), 'phoenix')
        self.assertEqual(data, image)
        self.assertEqual(post.call_args.kwargs['json_body'],
                         {'prompt': 'a cat', 'width': 1024, 'height': 1024,
                          'num_steps': 25})

    async def test_phoenix_json_base64_shape(self):
        # Other Workers AI models answer with {"result": {"image": ...}}.
        image = b'\xff\xd8\xffdata'
        body = ('{"result": {"image": "%s"}}'
                % base64.b64encode(image).decode()).encode()
        data, _ = await self._post(_response(body), 'phoenix')
        self.assertEqual(data, image)

    async def test_steps_out_of_range_is_rejected(self):
        for provider in image_gen.IMAGE_PROVIDERS:
            for steps in (0, 51):
                with self.assertRaises(ValueError):
                    await self._post(
                        _response(self._nvidia_body(b'x')), provider,
                        steps=steps)

    async def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            await image_gen.generate_image('a cat', provider='bogus')


class TestProviderCredentials(unittest.IsolatedAsyncioTestCase):
    # Every backend checks its config up front, not just the ones whose
    # first implementations happened to.

    async def test_missing_setting_is_reported(self):
        cases = [('nvidia-flux3', 'NVIDIA_API_KEY'),
                 ('nvidia-flux2', 'NVIDIA_API_KEY'),
                 ('phoenix', 'CLOUDFLARE_ACCOUNT_ID'),
                 ('phoenix', 'CLOUDFLARE_API_TOKEN'),
                 ('recraft', 'OPENROUTER_API_KEY')]
        for provider, attr in cases:
            with _configured(**{attr: ''}):
                with self.assertRaises(RuntimeError) as ctx:
                    await image_gen.generate_image('a cat',
                                                   provider=provider)
            self.assertIn(attr, str(ctx.exception))

class TestOpenRouterProvider(unittest.IsolatedAsyncioTestCase):
    # The OpenRouter images endpoint takes an aspect ratio rather than a
    # resolution, accepts n=1-6, and answers with inline base64 images.

    IMAGE = b'\x89PNG\r\n\x1a\nor'
    URL = 'https://openrouter.ai/api/v1/images'

    def _body(self, images: list) -> bytes:
        return json.dumps({'data': images}).encode()

    async def _render(self, images: list | None = None) -> tuple[
            bytes, unittest.mock.AsyncMock]:
        if images is None:
            images = [{'b64_json': base64.b64encode(self.IMAGE).decode()}]
        post = unittest.mock.AsyncMock(
            return_value=_response(self._body(images)))
        with _configured(), \
             unittest.mock.patch.object(http_client, 'post', post):
            data = await image_gen.generate_image('a cat',
                                                  provider='recraft')
        return data, post

    async def test_payload_and_default_aspect_ratio(self):
        # n is fixed at 6 (the tool only keeps the first image) and the
        # default aspect ratio is 'auto'.
        data, post = await self._render()
        self.assertEqual(data, self.IMAGE)
        self.assertEqual(post.call_args.args[0], self.URL)
        payload = post.call_args.kwargs['json_body']
        self.assertEqual(
            payload,
            {'model': 'recraft/recraft-v4.1-flash-v2:free',
             'prompt': 'a cat', 'response_format': 'b64_json',
             'aspect_ratio': 'auto', 'n': 6})

    async def test_explicit_aspect_ratio_is_sent(self):
        _, post = await self._render_post(aspect_ratio='16:9')
        self.assertEqual(post.call_args.kwargs['json_body']['aspect_ratio'],
                         '16:9')

    async def _render_post(self, **kwargs) -> tuple[
            bytes, unittest.mock.AsyncMock]:
        image = self.IMAGE
        post = unittest.mock.AsyncMock(return_value=_response(
            self._body([{'b64_json': base64.b64encode(image).decode()}])))
        with _configured(), \
             unittest.mock.patch.object(http_client, 'post', post):
            data = await image_gen.generate_image(
                'a cat', provider='recraft', **kwargs)
        return data, post

    async def test_first_of_several_images_is_kept(self):
        other = b'\x89PNG\r\n\x1a\nsecond'
        data, _ = await self._render(
            images=[{'b64_json': base64.b64encode(self.IMAGE).decode()},
                    {'b64_json': base64.b64encode(other).decode()}])
        self.assertEqual(data, self.IMAGE)

    async def test_no_images_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            await self._render(images=[])
        self.assertIn('no images', str(ctx.exception))

    async def test_missing_b64_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            await self._render(images=[{}])
        self.assertIn('no b64_json image data', str(ctx.exception))

    async def test_resolution_is_not_a_parameter(self):
        # The tool result reports the mismatch instead of silently ignoring
        # the resolution; no request is made.
        result = await image_gen.generate_and_send_image(
            mock_ctx(), 'a cat', provider='recraft', resolution='1024x1024')
        assert isinstance(result, str)
        self.assertIn("does not support the 'resolution' parameter", result)
        self.assertIn('aspect_ratio', result)

    async def test_invalid_aspect_ratio_is_reported(self):
        result = await image_gen.generate_and_send_image(
            mock_ctx(), 'a cat', provider='recraft', aspect_ratio='21:9')
        assert isinstance(result, str)
        self.assertIn("Invalid aspect_ratio '21:9'", result)

    async def test_valid_aspect_ratio_reaches_the_render(self):
        seen = {}

        async def fake_generate(prompt, width=1024, height=1024,
                                steps=None, provider='unset',
                                aspect_ratio=None):
            seen['aspect_ratio'] = aspect_ratio
            return b'\x89PNG\r\n\x1a\n'

        async def fake_store(user_id, data, name):
            return '/workspace/uploads/generated_1.png', None

        with unittest.mock.patch.object(image_gen, 'generate_image',
                                        fake_generate):
            with unittest.mock.patch.object(Messaging, 'send_image',
                                            unittest.mock.AsyncMock()):
                with unittest.mock.patch.object(image_gen.manager,
                                                'store_file', fake_store):
                    await image_gen.generate_and_send_image(
                        mock_ctx(), 'a cat', provider='recraft',
                        aspect_ratio='9:16')
        self.assertEqual(seen.get('aspect_ratio'), '9:16')

    async def test_no_resolution_still_works_for_every_provider(self):
        # resolution is now optional everywhere: the resolution-driven
        # backends still default to 1024x1024.
        image = b'\x89PNG\r\n\x1a\n'
        body = ('{"artifacts": [%s]}' % json.dumps(
            {'base64': base64.b64encode(image).decode(),
             'finishReason': 'SUCCESS'})).encode()
        post = unittest.mock.AsyncMock(return_value=_response(body))
        with _configured(), \
             unittest.mock.patch.object(http_client, 'post', post):
            data = await image_gen.generate_image('a cat')
        self.assertEqual(data, image)
        self.assertEqual(post.call_args.kwargs['json_body'],
                         {'prompt': 'a cat', 'width': 1024, 'height': 1024,
                          'steps': 50})


class TestGenerationDeadline(unittest.IsolatedAsyncioTestCase):
    # No single generation may run unbounded: the deadline covers the whole
    # render (submit, polling and any download together).

    async def test_render_overrunning_the_deadline_is_stopped(self):
        async def slow_post(*args, **kwargs):
            await asyncio.sleep(5)
            raise AssertionError('should have been cancelled')

        with _configured(), \
             unittest.mock.patch.object(image_gen,
                                        'IMAGE_GENERATION_DEADLINE', 0.05), \
             unittest.mock.patch.object(http_client, 'post', slow_post):
            with self.assertRaises(RuntimeError) as ctx:
                await image_gen.generate_image('a cat',
                                               provider='nvidia-flux3')
        self.assertIn('timed out', str(ctx.exception))


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

    def test_unknown_container_raises(self):
        # Guessing PNG for unknown bytes would store undeliverable files
        # under a lying extension, so it is an explicit error instead.
        for junk in (b'not an image at all', b'\x89PNG'):
            with self.assertRaises(ValueError):
                image_gen._sniff_format(junk)

    async def test_unknown_container_reported_to_the_model(self):
        async def fake_generate(*args, **_):
            return b'not an image'

        with unittest.mock.patch.object(image_gen, 'generate_image',
                                        fake_generate):
            result = await image_gen.generate_and_send_image(
                mock_ctx(), 'a cat')
        assert isinstance(result, str)
        self.assertIn('unsupported image format', result)

    async def test_webp_stored_with_webp_extension_and_mime(self):
        async def fake_generate(*args, **_):
            return b'RIFF\x00\x00\x00\x00WEBPdata'

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
            return b'RIFF\x00\x00\x00\x00WEBPdata'

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
