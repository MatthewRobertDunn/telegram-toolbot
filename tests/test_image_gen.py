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
    AI_HORDE_KEY='horde-key',
    AI_HORDE_API_URL='https://horde.test/api/v2',
    CLOUDFLARE_ACCOUNT_ID='acct',
    CLOUDFLARE_API_TOKEN='tok')


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
                        mock_ctx(), 'a cat', provider='horde')
        self.assertEqual(seen.get('provider'), 'horde')


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

    async def test_cloudflare_defaults_to_25_steps(self):
        # Phoenix takes the step count as num_steps; without a JSON content
        # type the raw body bytes are the image.
        image = b'\xff\xd8\xffdata'
        data, post = await self._post(
            _response(image, 'image/jpeg'), 'cloudflare')
        self.assertEqual(data, image)
        self.assertEqual(post.call_args.kwargs['json_body'],
                         {'prompt': 'a cat', 'width': 1024, 'height': 1024,
                          'num_steps': 25})

    async def test_cloudflare_json_base64_shape(self):
        # Other Workers AI models answer with {"result": {"image": ...}}.
        image = b'\xff\xd8\xffdata'
        body = ('{"result": {"image": "%s"}}'
                % base64.b64encode(image).decode()).encode()
        data, _ = await self._post(_response(body), 'cloudflare')
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
                 ('horde', 'AI_HORDE_KEY'),
                 ('cloudflare', 'CLOUDFLARE_ACCOUNT_ID'),
                 ('cloudflare', 'CLOUDFLARE_API_TOKEN')]
        for provider, attr in cases:
            with _configured(**{attr: ''}):
                with self.assertRaises(RuntimeError) as ctx:
                    await image_gen.generate_image('a cat',
                                                   provider=provider)
            self.assertIn(attr, str(ctx.exception))

class TestHordeProvider(unittest.IsolatedAsyncioTestCase):
    # Submit + poll + fetch: every wait is an await, so the shared event
    # loop is never blocked while a generation is queued.

    IMAGE = b'RIFF\x00\x00\x00\x00WEBPdata'

    async def _render(self, *, checks: tuple[dict, ...] = ({'finished': 1},),
                      generations: list | None = None,
                      extra_gets: tuple = ()) -> tuple[
            bytes, unittest.mock.AsyncMock, unittest.mock.AsyncMock]:
        if generations is None:
            generations = [{'img': 'data:image/webp;base64,'
                            + base64.b64encode(self.IMAGE).decode()}]
        get_responses = [_response(json.dumps(c).encode())
                         for c in checks]
        get_responses.append(
            _response(json.dumps({'generations': generations}).encode()))
        get_responses.extend(extra_gets)
        post = unittest.mock.AsyncMock(
            return_value=_response(b'{"id": "req-1"}'))
        get = unittest.mock.AsyncMock(side_effect=get_responses)
        with _configured(), \
             unittest.mock.patch.object(http_client, 'post', post), \
             unittest.mock.patch.object(http_client, 'get', get):
            data = await image_gen.generate_image('a cat', provider='horde')
        return data, post, get

    async def test_submit_payload_and_awaited_polling(self):
        # Not finished on the first check: the poll interval is awaited
        # (shortened here) rather than spun on a blocking request.
        with unittest.mock.patch.object(image_gen, 'HORDE_POLL_INTERVAL', 0):
            data, post, get = await self._render(
                checks=({'finished': 0}, {'finished': 1}))
        self.assertEqual(data, self.IMAGE)
        self.assertEqual(post.call_args.args[0],
                         _CREDENTIALS['AI_HORDE_API_URL'] + '/generate/async')
        payload = post.call_args.kwargs['json_body']
        self.assertEqual(payload['params']['steps'], 25)
        self.assertEqual(payload['params']['n'], 1)
        self.assertEqual(payload['nsfw'], False)
        # Three GETs: two checks plus the status fetch.
        self.assertEqual(get.await_count, 3)

    async def test_image_served_from_a_url_is_downloaded(self):
        image_get = _response(self.IMAGE, 'image/webp')
        data, post, get = await self._render(
            generations=[{'img': 'https://r2.test/img.webp'}],
            extra_gets=(image_get,))
        self.assertEqual(data, self.IMAGE)
        assert get.await_args is not None
        self.assertEqual(get.await_args.args[0], 'https://r2.test/img.webp')

    async def test_faulted_generation_raises(self):
        with self.assertRaises(RuntimeError):
            await self._render(checks=({'finished': 0, 'faulted': True},))

    async def test_no_generations_raises(self):
        with self.assertRaises(RuntimeError):
            await self._render(generations=[])

    async def test_malformed_data_uri_is_reported(self):
        # 'data:' without a comma would previously raise a bare IndexError.
        with self.assertRaises(RuntimeError) as ctx:
            await self._render(generations=[{'img': 'data:image/webp'}])
        self.assertIn('malformed', str(ctx.exception))

    async def test_http_error_from_the_endpoint_carries_the_body(self):
        post = unittest.mock.AsyncMock(
            return_value=_response(b'{"message": "not enough kudos"}',
                                        status=429))
        with _configured(), \
             unittest.mock.patch.object(http_client, 'post', post):
            with self.assertRaises(http_client.HttpStatusError):
                await image_gen.generate_image('a cat', provider='horde')


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
