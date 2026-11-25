import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
import cachetools
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sandbox_image', 'helpers'))

import toolbot_edit
import toolbot_glob
import toolbot_grep
import toolbot_protocol
import toolbot_read
from handlers import upload_handler as upload_handler_mod
from handlers import run_llm as run_llm_mod
from conversation import Conversation
from mock_context import mock_ctx
from tools.sandbox import common, files, media, search, shell
from tools import python_runner
from sandbox import SandboxManager
from tool_result import ImageAttachment, ToolCallResult
from tool_registry import ToolSurface
import upload_cache



# Modules holding a `manager` reference to patch; search.py goes through
# common._run_helper, so it has none.
_SANDBOX_MODULES = (common, shell, files, media, python_runner)


@contextlib.contextmanager
def _patched_manager(manager):
    with contextlib.ExitStack() as stack:
        for module in _SANDBOX_MODULES:
            stack.enter_context(
                unittest.mock.patch.object(module, "manager", manager))
        yield


class _FakeBot:
    def __init__(self):
        self.sent = []
        self.sent_photos = []

    async def send_document(self, *args, **kwargs):
        document = args[1] if len(args) >= 2 else kwargs.get("document")
        self.sent.append(
            {"filename": kwargs.get("filename"), "document": document})

    async def send_photo(self, *args, **kwargs):
        photo = args[1] if len(args) >= 2 else kwargs.get("photo")
        self.sent_photos.append(photo)


class CannedDocker:
    """Fake docker CLI with scripted exec responses.

    exec responses are matched in order against the command string; the
    first matching (substring, returncode, stdout, stderr) tuple is used.
    inspect/run/rm/ps/start/stop behave like the FakeDocker in test_sandbox.
    """

    def __init__(self, exec_responses: list[tuple[str, int, str, str]]):
        self.exec_responses = exec_responses
        self.calls: list[list[str]] = []
        self.containers: dict[str, str] = {}

    async def __call__(self, args: list[str], timeout=None, input=None):
        self.calls.append(list(args))
        verb = args[0]
        if verb == 'inspect':
            name = args[3]
            if name in self.containers:
                return 0, self.containers[name] + '\n', ''
            return 1, '', 'No such object'
        if verb == 'run':
            name = args[args.index('--name') + 1]
            self.containers[name] = 'true'
            return 0, f'{name}-id\n', ''
        if verb == 'start':
            name = args[1]
            if name not in self.containers:
                return 1, '', 'No such container'
            self.containers[name] = 'true'
            return 0, name + '\n', ''
        if verb == 'stop':
            name = args[1]
            if name not in self.containers:
                return 1, '', 'No such container'
            self.containers[name] = 'false'
            return 0, name + '\n', ''
        if verb == 'rm':
            name = args[2]
            self.containers.pop(name, None)
            return 0, name, ''
        if verb == 'exec':
            command = args[-1]
            for substring, rc, stdout, stderr in self.exec_responses:
                if substring in command:
                    return rc, stdout, stderr
            return 0, '', ''
        return 0, '', ''


def _run_main(argv: list[str], fn) -> tuple[dict, int]:
    """Run a helper's main() with argv; returns (stdout payload, exit code).

    Helpers exit 0 only on success (toolbot_protocol.py).
    """
    original = sys.argv
    sys.argv = ['helper'] + argv
    captured = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(captured):
            code = fn() or 0
    finally:
        sys.argv = original
    return json.loads(captured.getvalue()), code


class TestEditHelper(unittest.TestCase):

    def _replace(self, content, old, new, replace_all=False):
        return toolbot_edit.replace(content, old, new, replace_all)

    def _run(self, payload_text: str) -> tuple[dict, int]:
        original_stdin = sys.stdin
        sys.stdin = io.StringIO(payload_text)
        try:
            return _run_main([], toolbot_edit.main)
        finally:
            sys.stdin = original_stdin

    def test_exact_replacement(self):
        self.assertEqual(self._replace('foo\nbar\nbaz\n', 'bar', 'qux'),
                         'foo\nqux\nbaz\n')

    def test_not_found_or_ambiguous_is_refused(self):
        with self.assertRaises(ValueError, msg='Could not find oldString'):
            self._replace('foo\n', 'nope', 'x')
        with self.assertRaises(ValueError, msg='Found multiple matches'):
            self._replace('x = 1\ny = x\nx = 2\n', 'x', 'z')

    def test_replace_all(self):
        self.assertEqual(self._replace('a b\na c\n', 'a', 'z', True),
                         'z b\nz c\n')

    def test_unique_context_disambiguates(self):
        content = 'def a():\n    return 1\n\ndef b():\n    return 1\n'
        old = 'def b():\n    return 1'
        new = 'def b():\n    return 2'
        self.assertEqual(self._replace(content, old, new),
                         'def a():\n    return 1\n\ndef b():\n    return 2\n')

    def test_line_trimmed_fallback(self):
        content = 'def f():\n\treturn 1\n'
        old = 'def f():\n    return 1'
        result = self._replace(content, old, 'def f():\n\treturn 2')
        self.assertEqual(result, 'def f():\n\treturn 2\n')

        # The fallback also flexes block indentation.
        content = 'class A:\n    def m(self):\n        pass\n'
        old = 'def m(self):\npass'
        result = self._replace(content, old, 'def m(self):\n    return 5')
        self.assertIn('return 5', result)

    def test_disproportionate_match_rejected(self):
        content = 'line\n' * 20
        with self.assertRaises(ValueError, msg='Refusing replacement'):
            self._replace(content, 'line', 'x' * 2000)

    def test_line_endings_preserved(self):
        content = 'foo\r\nbar\r\n'
        result = self._replace(content, 'bar', 'baz')
        self.assertEqual(result, 'foo\r\nbaz\r\n')

    def test_helper_script_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.txt')
            with open(path, 'w') as f:
                f.write('foo\nbar\n')
            payload = {'path': path, 'old_string': 'bar', 'new_string': 'qux',
                       'replace_all': False}
            result, code = self._run(json.dumps(payload))
            self.assertEqual(code, 0)
            self.assertTrue(result['ok'])
            self.assertEqual(result['v'], toolbot_protocol.PROTOCOL_VERSION)
            with open(path) as f:
                self.assertEqual(f.read(), 'foo\nqux\n')

    def test_payload_rejections(self):
        cases = [
            (json.dumps({'path': 'f.txt', 'old_string': 'a',
                         'new_string': 'b', 'extra': 1}),
             'unexpected payload field'),
            (json.dumps({'old_string': 'a', 'new_string': 'b'}), 'payload'),
            (json.dumps({'path': 3, 'old_string': 'a',
                         'new_string': 'b'}), 'payload'),
            (json.dumps({'path': 'f.txt', 'new_string': 'b'}), 'payload'),
            (json.dumps({'path': 'f.txt', 'old_string': 'a',
                         'new_string': 'b', 'replace_all': 'yes'}), 'payload'),
            ('not json at all', 'not valid JSON'),
            ('[1, 2]', 'JSON object'),
        ]
        for payload, keyword in cases:
            with self.subTest(payload=payload):
                result, code = self._run(payload)
                self.assertEqual(code, 1)
                self.assertFalse(result['ok'])
                self.assertIn(keyword, result['error'])


class TestReadHelper(unittest.TestCase):

    def _read(self, *args) -> tuple[dict, int]:
        return _run_main(list(args), toolbot_read.main)

    def test_line_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.txt')
            with open(path, 'w') as f:
                f.write('a\nb\nc\n')
            result, code = self._read('--path', path)
            self.assertEqual(code, 0)
            self.assertTrue(result['ok'])
            data = result['data']
            self.assertEqual(data['type'], 'file')
            self.assertEqual([line['text'] for line in data['lines']],
                             ['a', 'b', 'c'])
            self.assertEqual([line['n'] for line in data['lines']], [1, 2, 3])
            self.assertEqual(data['totalLines'], 3)
            self.assertFalse(data['more'])
            self.assertFalse(data['cut'])

    def test_offset_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.txt')
            with open(path, 'w') as f:
                f.writelines(f'{i}\n' for i in range(1, 11))
            result, code = self._read('--path', path, '--offset', '5',
                                      '--limit', '3')
            self.assertEqual(code, 0)
            data = result['data']
            self.assertEqual([line['n'] for line in data['lines']], [5, 6, 7])
            self.assertTrue(data['more'])

    def test_offset_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.txt')
            with open(path, 'w') as f:
                f.write('a\n')
            result, code = self._read('--path', path, '--offset', '10')
            self.assertEqual(code, 1)
            self.assertFalse(result['ok'])
            self.assertIn('out of range', result['error'])

    def test_long_line_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.txt')
            with open(path, 'w') as f:
                f.write('x' * 3000 + '\n')
            result, _ = self._read('--path', path)
            text = result['data']['lines'][0]['text']
            self.assertEqual(
                len(text), 2000 + len(toolbot_read.MAX_LINE_SUFFIX))
            self.assertTrue(text.endswith('chars)'))

    def test_directory_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'sub'))
            with open(os.path.join(tmp, 'file.txt'), 'w') as f:
                f.write('x')
            result, code = self._read('--path', tmp)
            self.assertEqual(code, 0)
            data = result['data']
            self.assertEqual(data['type'], 'directory')
            self.assertEqual(data['entries'], ['file.txt', 'sub/'])
            self.assertEqual(data['totalEntries'], 2)
            self.assertFalse(data['truncated'])

    def test_binary_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.zip')
            with open(path, 'wb') as f:
                f.write(b'PK\x03\x04' + b'\x00' * 100)
            result, code = self._read('--path', path)
            self.assertEqual(code, 1)
            self.assertFalse(result['ok'])
            self.assertIn('binary', result['error'])

    def test_not_found_with_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'app.py'), 'w') as f:
                f.write('x')
            result, code = self._read('--path', os.path.join(tmp, 'app'))
            self.assertEqual(code, 1)
            self.assertFalse(result['ok'])
            self.assertIn('Did you mean', result['error'])
            self.assertIn('app.py', result['error'])

        with tempfile.TemporaryDirectory() as tmp:
            result, code = self._read('--path', os.path.join(tmp, 'missing.txt'))
            self.assertEqual(code, 1)
            self.assertFalse(result['ok'])
            self.assertIn('File not found', result['error'])


class TestGlobHelper(unittest.TestCase):

    def setUp(self):
        self._original_rg = toolbot_glob._rg_files

    def tearDown(self):
        toolbot_glob._rg_files = self._original_rg

    def test_sorted_by_mtime_descending(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, mtime in (('old.txt', 1000), ('new.txt', 2000)):
                path = os.path.join(tmp, name)
                with open(path, 'w') as f:
                    f.write('x')
                os.utime(path, (mtime, mtime))
            toolbot_glob._rg_files = (
                lambda path, pattern:
                [os.path.join(tmp, 'old.txt'), os.path.join(tmp, 'new.txt')])
            result = toolbot_glob.find_files(tmp, '*.txt')
            self.assertEqual(
                result['files'],
                [os.path.join(tmp, 'new.txt'), os.path.join(tmp, 'old.txt')])
            self.assertFalse(result['truncated'])

    def test_truncation_cap(self):
        toolbot_glob._rg_files = (
            lambda path, pattern: [f'f{i}.txt' for i in range(150)])
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(150):
                with open(os.path.join(tmp, f'f{i}.txt'), 'w') as f:
                    f.write('x')
            result = toolbot_glob.find_files(tmp, '*.txt')
            self.assertEqual(len(result['files']), toolbot_glob.LIMIT)
            self.assertTrue(result['truncated'])

    def test_path_must_be_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, 'f.txt')
            with open(file_path, 'w') as f:
                f.write('x')
            result, code = _run_main(
                ['--pattern', '*', '--path', file_path], toolbot_glob.main)
            self.assertEqual(code, 1)
            self.assertFalse(result['ok'])
            self.assertIn('directory', result['error'])


class TestGrepHelper(unittest.TestCase):
    """Mirrors TestGlobHelper: search() runs against a stubbed rg seam."""

    def setUp(self):
        self._original_rg = toolbot_grep._start_rg

    def tearDown(self):
        toolbot_grep._start_rg = self._original_rg

    @staticmethod
    def _fake_rg(events: list, returncode: int = 0):
        class _Proc:
            stdout = iter([event + '\n' for event in events])

            def __init__(self):
                self.killed = False
                self.returncode = None

            def kill(self):
                self.killed = True

            def wait(self):
                self.returncode = returncode
                return returncode

        return _Proc()

    @staticmethod
    def _match_event(path: str, line: int, text: str) -> str:
        return json.dumps({'type': 'match', 'data': {
            'path': {'text': path}, 'line_number': line,
            'lines': {'text': text}}})

    def test_matches_are_parsed_and_made_absolute(self):
        events = [
            json.dumps({'type': 'begin', 'data': {'path': {'text': 'a.py'}}}),
            self._match_event('a.py', 3, 'needle here\n'),
            self._match_event('a.py', 7, 'another needle\n'),
            json.dumps({'type': 'end', 'data': {}}),
        ]
        toolbot_grep._start_rg = (lambda path, pattern, include:
                                  self._fake_rg(events))
        result = toolbot_grep.search('/workspace', 'needle', None)
        self.assertEqual(result['matches'], [
            {'path': os.path.abspath('a.py'), 'line': 3,
             'text': 'needle here'},
            {'path': os.path.abspath('a.py'), 'line': 7,
             'text': 'another needle'},
        ])
        self.assertEqual(result['totalMatches'], 2)
        self.assertFalse(result['truncated'])

    def test_non_json_lines_tolerated(self):
        toolbot_grep._start_rg = (lambda path, pattern, include:
                                  self._fake_rg(['partial-truncated-garbage',
                                                 self._match_event('a', 1, 'x\n')]))
        result = toolbot_grep.search('.', 'x', None)
        self.assertEqual(len(result['matches']), 1)

    def test_truncation_cap_and_early_stop(self):
        events = [self._match_event('a.py', i, 'hit\n')
                  for i in range(toolbot_grep.LIMIT + 10)]
        proc = self._fake_rg(events)
        toolbot_grep._start_rg = (lambda path, pattern, include: proc)
        result = toolbot_grep.search('.', 'hit', None)
        self.assertEqual(len(result['matches']), toolbot_grep.LIMIT)
        self.assertEqual(result['totalMatches'], toolbot_grep.LIMIT + 1)
        self.assertTrue(result['truncated'])
        self.assertTrue(proc.killed)

    def test_long_lines_capped(self):
        toolbot_grep._start_rg = (lambda path, pattern, include:
                                  self._fake_rg(
                                      [self._match_event('a.py', 1,
                                                         'x' * 3000 + '\n')]))
        result = toolbot_grep.search('.', 'x', None)
        self.assertEqual(
            len(result['matches'][0]['text']),
            toolbot_grep.MAX_LINE_CHARS + len('... (line truncated)'))

    def test_rg_failure_after_clean_read(self):
        toolbot_grep._start_rg = (lambda path, pattern, include:
                                  self._fake_rg([], returncode=2))
        with self.assertRaises(RuntimeError, msg='ripgrep failed'):
            toolbot_grep.search('.', 'x', None)

    def test_path_must_be_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, 'f.txt')
            with open(file_path, 'w') as f:
                f.write('x')
            result, code = _run_main(
                ['--pattern', 'x', '--path', file_path],
                toolbot_grep.main)
            self.assertEqual(code, 1)
            self.assertFalse(result['ok'])
            self.assertIn('directory', result['error'])


_MEDIA_ATTRS = ('document', 'photo', 'audio', 'video', 'voice',
                'video_note', 'animation', 'sticker')


class TestUploadHandler(unittest.IsolatedAsyncioTestCase):
    """The upload handler stores documents/photos in the sandbox and
    queues a generation priming the model with the stored path."""

    def _ctx(self, file_path='/workspace/uploads/f.bin', **media_attrs):
        bot = _UploadBot(file_path)
        ctx = mock_ctx(user_id=7, bot=bot)
        ctx.message.message_id = 42
        # A Telegram message always carries every media attribute, empty
        # when there is no upload; _get_media dispatches on exactly those.
        for attr in _MEDIA_ATTRS:
            setattr(ctx.message, attr, None)
        setattr(ctx.message, 'photo', ())
        for attr, value in media_attrs.items():
            setattr(ctx.message, attr, value)
        return ctx, bot

    @contextlib.contextmanager
    def _env(self, store_return=('/workspace/uploads/pic.png', None)):
        store = unittest.mock.AsyncMock(return_value=store_return)
        submit = _FakeSubmit()
        recorded = _FakeRecord()
        with unittest.mock.patch.object(
                upload_handler_mod, 'manager',
                SimpleNamespace(store_file=store)):
            with unittest.mock.patch.object(
                    upload_handler_mod, 'submit_generation', submit):
                with unittest.mock.patch.object(
                        upload_handler_mod, 'record_upload', recorded):
                    yield store, submit, recorded

    async def test_document_upload_queues_a_generation(self):
        doc = SimpleNamespace(file_id='f1', file_name='pic.png')
        ctx, bot = self._ctx(document=doc)
        with self._env() as (store, submit, recorded):
            await upload_handler_mod.upload_handler(ctx)
        self.assertIsNotNone(submit.captured)
        store.assert_awaited_once()
        self.assertEqual(store.await_args_list[0].args,
                         (7, b'file-bytes', 'pic.png'))
        self.assertEqual(len(submit.calls), 1)
        self.assertEqual(
            submit.captured['content'],
            '[The user uploaded a file to their sandbox: '
            '/workspace/uploads/pic.png (10 bytes). You can inspect it '
            'with the read_image tool if it is relevant.]')
        self.assertEqual(submit.captured['kwargs']['telegram_message_id'], 42)
        self.assertEqual(recorded.calls, [(7, '/workspace/uploads/pic.png')])

    async def test_caption_prefixes_the_notification(self):
        doc = SimpleNamespace(file_id='f1', file_name='note.txt')
        ctx, bot = self._ctx(document=doc)
        ctx.message.caption = 'look at this'
        with self._env(('/workspace/uploads/note.txt', None)) as (
                store, submit, recorded):
            await upload_handler_mod.upload_handler(ctx)
        self.assertEqual(
            submit.captured['content'],
            'look at this\n\n[The user uploaded a file to their sandbox: '
            '/workspace/uploads/note.txt (10 bytes). You can inspect it with '
            'the read tool if it is relevant.]')

    async def test_photo_without_a_name_gets_a_fallback_name(self):
        # Telegram photos arrive without file_name; the handler derives
        # photo_<epoch><ext> from the media kind and the download's path.
        photo = PhotoSize(file_id='f2')
        ctx, bot = self._ctx('/workspace/uploads/x.jpg', photo=[photo])
        with self._env(('/workspace/uploads/photo_1.jpg', None)) as (
                store, submit, recorded):
            await upload_handler_mod.upload_handler(ctx)
        store.assert_awaited_once()
        name = store.await_args_list[0].args[2]
        self.assertTrue(name.startswith('photo_'), name)
        self.assertTrue(name.endswith('.jpg'), name)
        self.assertEqual(bot.last_file_id, 'f2')

    async def test_no_media_is_ignored(self):
        ctx, bot = self._ctx()
        with self._env() as (store, submit, recorded):
            await upload_handler_mod.upload_handler(ctx)
        store.assert_not_awaited()
        self.assertEqual(submit.calls, [])
        self.assertEqual(recorded.calls, [])
        self.assertEqual(bot.sent, [])

    async def test_download_failure_is_reported(self):
        ctx, bot = self._ctx(document=SimpleNamespace(file_id='f1'))
        bot.fail_download = True
        with self._env() as (store, submit, recorded):
            await upload_handler_mod.upload_handler(ctx)
        store.assert_not_awaited()
        self.assertEqual(submit.calls, [])
        self.assertEqual(len(bot.sent), 1)
        self.assertIn('Error processing file', bot.sent[0])

    async def test_store_failure_is_reported(self):
        ctx, bot = self._ctx(document=SimpleNamespace(file_id='f1'))
        with self._env((None, 'disk full')) as (store, submit, recorded):
            await upload_handler_mod.upload_handler(ctx)
        self.assertEqual(submit.calls, [])
        self.assertEqual(recorded.calls, [])
        self.assertEqual(bot.sent, ['disk full'])


class PhotoSize(SimpleNamespace):
    """Minimal stand-in for telegram.PhotoSize (name drives the fallback)."""


class _UploadBot:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fail_download = False
        self.last_file_id = None
        self.sent = []

    async def get_file(self, file_id):
        self.last_file_id = file_id
        return _UploadFile(self, file_id)

    async def send_message(self, *args, **kwargs):
        self.sent.append(args[0] if args else kwargs.get('text'))


class _UploadFile:
    def __init__(self, bot, file_id):
        self._bot = bot
        self.file_path = bot.file_path

    async def download_as_bytearray(self):
        if self._bot.fail_download:
            raise RuntimeError('telegram is down')
        return bytearray(b'file-bytes')


class _FakeSubmit:
    def __init__(self):
        self.calls = []
        self.captured: dict = {}

    async def __call__(self, ctx, content, **kwargs):
        self.calls.append((ctx, content))
        self.captured = dict(ctx=ctx, content=content, kwargs=kwargs)


class _FakeRecord:
    def __init__(self):
        self.calls = []

    def __call__(self, user_id, path):
        self.calls.append((user_id, path))


class TestUploadCache(unittest.TestCase):

    def setUp(self):
        self._original_cache = upload_cache._recent_uploads

    def tearDown(self):
        upload_cache._recent_uploads = self._original_cache

    def _swap_cache(self, ttl):
        now = [1000.0]
        upload_cache._recent_uploads = cachetools.TTLCache(
            maxsize=8, ttl=ttl, timer=lambda: now[0])
        return now

    def test_matches_by_user_and_path(self):
        self._swap_cache(60)
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        self.assertTrue(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/pic.png'))
        # Matching is on the full path, so a same-named file elsewhere
        # does not collide with it.
        self.assertFalse(upload_cache.was_recently_uploaded(1, 'uploads/pic.png'))
        self.assertFalse(
            upload_cache.was_recently_uploaded(2, '/workspace/uploads/pic.png'))
        self.assertFalse(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/other.png'))

    def test_entries_expire_after_ttl(self):
        now = self._swap_cache(60)
        upload_cache.record_upload(1, '/workspace/uploads/pic.png')
        now[0] += 60  # exactly at expiry
        self.assertFalse(
            upload_cache.was_recently_uploaded(1, '/workspace/uploads/pic.png'))


class TestHelperRunner(unittest.IsolatedAsyncioTestCase):
    def _manager(self, rc: int, stdout: str, stderr: str = ''):
        return SandboxManager(docker=CannedDocker([('helper.py', rc, stdout,
                                                    stderr)]))

    async def _run(self, manager) -> dict:
        with _patched_manager(manager):
            return await common._run_helper(1, 'helper.py', ['--flag'])

    async def test_valid_success_response(self):
        payload = json.dumps({'v': 1, 'ok': True, 'data': {'x': 1}})
        result = await self._run(self._manager(0, payload + '\n'))
        self.assertEqual(result, {'v': 1, 'ok': True, 'data': {'x': 1}})

    async def test_valid_failure_response(self):
        payload = json.dumps({'v': 1, 'ok': False, 'error': 'boom'})
        result = await self._run(self._manager(1, payload + '\n'))
        self.assertEqual(result, {'ok': False, 'error': 'boom'})

    async def test_nonzero_exit_reports_stderr(self):
        result = await self._run(self._manager(2, '', 'traceback text'))
        self.assertFalse(result['ok'])
        self.assertEqual(result['error'], 'traceback text')

    async def test_nonzero_exit_without_stderr(self):
        result = await self._run(self._manager(2, '', '  \n'))
        self.assertFalse(result['ok'])
        self.assertIn('exited with code 2', result['error'])

    async def test_success_payload_on_nonzero_exit_reports_the_run(self):
        # A "success" payload arriving with a failed exit is untrustworthy:
        # the run's exit status is what gets reported, not the payload.
        payload = json.dumps({'v': 1, 'ok': True, 'data': {'x': 1}})
        result = await self._run(self._manager(3, payload, 'died badly'))
        self.assertFalse(result['ok'])
        self.assertEqual(result['error'], 'died badly')

        result = await self._run(self._manager(3, payload))
        self.assertEqual(result['error'], 'helper.py exited with code 3.')

    async def test_missing_version_read_as_v1(self):
        # No "v" field reads as v1, so sandbox images deployed without
        # the version field keep working.
        payload = json.dumps({'ok': True, 'data': {'x': 1}})
        result = await self._run(self._manager(0, payload + '\n'))
        self.assertEqual(result, {'ok': True, 'data': {'x': 1}})

        payload = json.dumps({'ok': False, 'error': 'old helper failure'})
        result = await self._run(self._manager(1, payload + '\n'))
        self.assertEqual(result, {'ok': False, 'error': 'old helper failure'})

    async def test_versions_up_to_max_accepted(self):
        for version in (1, 2):
            payload = json.dumps(
                {'v': version, 'ok': True, 'data': {'x': version}})
            result = await self._run(self._manager(0, payload))
            self.assertEqual(result['data'], {'x': version})

    async def test_future_protocol_version_rejected(self):
        payload = json.dumps({'v': common.MAX_HELPER_PROTOCOL_VERSION + 1,
                              'ok': True, 'data': {}})
        result = await self._run(self._manager(0, payload))
        self.assertFalse(result['ok'])
        self.assertIn('newer than this bot understands', result['error'])

    async def test_missing_ok_flag(self):
        payload = json.dumps({'v': 1, 'data': {}})
        result = await self._run(self._manager(0, payload))
        self.assertFalse(result['ok'])
        self.assertIn("'ok' flag", result['error'])

    async def test_success_without_data(self):
        payload = json.dumps({'v': 1, 'ok': True})
        result = await self._run(self._manager(0, payload))
        self.assertFalse(result['ok'])
        self.assertIn("missing 'data'", result['error'])

    async def test_failure_without_error_message(self):
        payload = json.dumps({'v': 1, 'ok': False})
        result = await self._run(self._manager(1, payload))
        self.assertFalse(result['ok'])
        self.assertIn('without an error message', result['error'])


class TestSandboxTools(unittest.IsolatedAsyncioTestCase):

    def _canned(self, exec_responses):
        return SandboxManager(docker=CannedDocker(exec_responses))

    async def test_command_tools_share_formatting(self):
        # Both command tools render through the one formatter, so the model
        # sees the same shape from either entry point.
        cases = [
            (shell.bash, 'echo hello', ('echo', 0, 'hello\n', ''),
             ['Return code: 0', 'stdout:\nhello'], None),
            (python_runner.run_python_code, 'print(2+2)',
             ('python3 -', 0, '4\n', ''),
             [], 'Return code: 0\nstdout:\n4\n'),
            (python_runner.run_python_code, 'def def def',
             ('python3 -', 1, '', 'SyntaxError: bad code'),
             [], 'Return code: 1\nstderr:\nSyntaxError: bad code'),
        ]
        for tool, command, exec_response, substrings, exact in cases:
            with self.subTest(tool=tool.__name__, command=command):
                manager = self._canned([exec_response])
                with _patched_manager(manager):
                    result = await tool(mock_ctx(), command)
                for substring in substrings:
                    self.assertIn(substring, result)
                if exact is not None:
                    self.assertEqual(result, exact)

    async def test_write_creates_and_marks_read(self):
        manager = self._canned([('test -e', 1, '', ''), ('mkdir', 0, '', '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await files.write(ctx, 'src/new.txt', 'data')
        self.assertEqual(result, 'Wrote file successfully.')
        self.assertIn('/workspace/src/new.txt', ctx.user_data.sandbox_read_files)

    async def test_requires_read_first(self):
        # Writing an existing file and editing both demand a prior read.
        manager = self._canned([('test -e', 0, '', '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await files.write(ctx, 'existing.txt', 'data')
        self.assertIn('has not been read', result)

        manager = self._canned([])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await files.edit(ctx, 'f.txt', 'old', 'new')
        self.assertIn('has not been read', result)

    async def test_edit_runs_helper(self):
        helper_result = json.dumps(
            {'v': 1, 'ok': True, 'data': {'output': 'Edit applied successfully.'}})
        manager = self._canned([('toolbot_edit.py', 0, helper_result, '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            ctx.user_data.sandbox_read_files = {'/workspace/f.txt'}
            result = await files.edit(ctx, 'f.txt', 'old', 'new')
        self.assertEqual(result, 'Edit applied successfully.')

    async def test_edit_identical_strings(self):
        manager = self._canned([])
        with _patched_manager(manager):
            ctx = mock_ctx()
            ctx.user_data.sandbox_read_files = {'/workspace/f.txt'}
            result = await files.edit(ctx, 'f.txt', 'same', 'same')
        self.assertIn('identical', result)

    async def test_read_formats_lines(self):
        helper_result = json.dumps({'v': 1, 'ok': True, 'data': {
            'type': 'file', 'path': '/workspace/f.txt',
            'lines': [{'n': 1, 'text': 'alpha'}, {'n': 2, 'text': 'beta'}],
            'lineStart': 1, 'lineEnd': 2, 'totalLines': 2,
            'more': False, 'cut': False}})
        manager = self._canned([('toolbot_read.py', 0, helper_result, '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await files.read(ctx, 'f.txt')
        self.assertIn('1: alpha', result)
        self.assertIn('2: beta', result)
        self.assertIn('End of file - total 2 lines', result)
        self.assertIn('/workspace/f.txt', ctx.user_data.sandbox_read_files)

    async def test_read_error(self):
        helper_result = json.dumps(
            {'v': 1, 'ok': False, 'error': 'File not found: /workspace/x'})
        manager = self._canned([('toolbot_read.py', 0, helper_result, '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await files.read(ctx, 'x')
        self.assertIn('File not found', result)

    async def test_read_directory(self):
        helper_result = json.dumps({'v': 1, 'ok': True, 'data': {
            'type': 'directory', 'path': '/workspace',
            'entries': ['src/', 'app.py'], 'offset': 1,
            'totalEntries': 2, 'truncated': False}})
        manager = self._canned([('toolbot_read.py', 0, helper_result, '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await files.read(ctx, '.')
        self.assertIn('<type>directory</type>', result)
        self.assertIn('src/', result)
        self.assertNotIn('/workspace',
                         ctx.user_data.sandbox_read_files)

    async def test_grep_delegates_to_the_helper(self):
        helper_result = json.dumps({'v': 1, 'ok': True, 'data': {
            'matches': [
                {'path': '/workspace/src/a.py', 'line': 4, 'text': 'find_me()'},
                {'path': '/workspace/src/b.py', 'line': 9, 'text': 'find_me()'},
            ],
            'totalMatches': 2, 'truncated': False}})
        manager = self._canned([('toolbot_grep.py', 0, helper_result, '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await search.grep(ctx, 'find_me')
        self.assertIn('Found 2 matches', result)
        self.assertIn('/workspace/src/a.py:', result)
        self.assertIn('  Line 4: find_me()', result)
        self.assertIn('  Line 9: find_me()', result)
        # The pattern rides in as helper arguments; rg runs inside the image.
        self.assertIn('--pattern', manager._docker.calls[-1][-1])

        # An empty result set is a distinct contract the model re-queries on.
        helper_result = json.dumps(
            {'v': 1, 'ok': True, 'data': {'matches': [], 'totalMatches': 0,
                                          'truncated': False}})
        manager = self._canned([('toolbot_grep.py', 0, helper_result, '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await search.grep(ctx, 'nothing')
        self.assertEqual(result, 'No files found')

    async def test_grep_reports_helper_failures(self):
        cases = [
            (('toolbot_grep.py', 0,
              json.dumps({'v': 1, 'ok': False, 'error': 'regex parse error'}),
              ''), ['regex parse error']),
            # A helper that dies without speaking the protocol degrades to
            # an error; its traceback never reaches the model.
            (('toolbot_grep.py', 2, '',
              'Traceback (most recent call last): ...'),
             ['Traceback', 'Error:']),
        ]
        for exec_response, expected in cases:
            with self.subTest(exec_response=exec_response):
                manager = self._canned([exec_response])
                with _patched_manager(manager):
                    result = await search.grep(mock_ctx(), '[')
                for substring in expected:
                    self.assertIn(substring, result)

    async def test_grep_truncates(self):
        helper_result = json.dumps({'v': 1, 'ok': True, 'data': {
            'matches': [
                {'path': '/workspace/a.py', 'line': i, 'text': 'hit'}
                for i in range(100)],
            'totalMatches': 110, 'truncated': True}})
        manager = self._canned([('toolbot_grep.py', 0, helper_result, '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await search.grep(ctx, 'hit')
        self.assertIn('Found 100 matches (more matches available)', result)
        self.assertIn('Results truncated', result)

    async def test_glob_formats_files(self):
        helper_result = json.dumps({'v': 1, 'ok': True, 'data': {
            'files': ['/workspace/a.py', '/workspace/b.py'],
            'truncated': False}})
        manager = self._canned([('toolbot_glob.py', 0, helper_result, '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await search.glob(ctx, '*.py')
        self.assertIn('/workspace/a.py', result)
        self.assertIn('/workspace/b.py', result)

        helper_result = json.dumps(
            {'v': 1, 'ok': True, 'data': {'files': [], 'truncated': False}})
        manager = self._canned([('toolbot_glob.py', 0, helper_result, '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await search.glob(ctx, '*.nope')
        self.assertEqual(result, 'No files found')

    async def test_patch_success(self):
        summary = 'Success. Updated the following files:\nA src/new.py\nM app.py'
        manager = self._canned([('apply-patch-py', 0, summary + '\n', '')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await files.patch(
                ctx, '*** Begin Patch\n*** End Patch')
        self.assertIn('A src/new.py', result)
        self.assertIn('/workspace/src/new.py', ctx.user_data.sandbox_read_files)

    async def test_patch_error(self):
        manager = self._canned(
            [('apply-patch-py', 1, '', 'patch rejected: bad header')])
        with _patched_manager(manager):
            ctx = mock_ctx()
            result = await files.patch(ctx, 'garbage')
        self.assertIn('patch rejected', result)


class TestSendFile(unittest.IsolatedAsyncioTestCase):

    def _canned(self, exec_responses):
        return SandboxManager(docker=CannedDocker(exec_responses))

    async def test_send_file_success(self):
        encoded = base64.b64encode(b"hello world").decode()
        manager = self._canned([
            ('stat', 0, '11\n', ''),
            ('base64', 0, encoded + '\n', ''),
        ])
        bot = _FakeBot()
        with _patched_manager(manager):
            ctx = mock_ctx(bot=bot)
            result = await media.send_file(ctx, 'f.txt')
        self.assertEqual(result, 'Sent f.txt (11 bytes).')
        self.assertEqual(bot.sent[0]['filename'], 'f.txt')
        self.assertEqual(bot.sent[0]['document'], b'hello world')

        # Arbitrary binary payloads survive the base64 round trip.
        payload = bytes(range(256))
        encoded = base64.b64encode(payload).decode()
        manager = self._canned([
            ('stat', 0, f'{len(payload)}\n', ''),
            ('base64', 0, encoded, ''),
        ])
        bot = _FakeBot()
        with _patched_manager(manager):
            ctx = mock_ctx(bot=bot)
            await media.send_file(ctx, 'blob.bin')
        self.assertEqual(bot.sent[0]['document'], payload)

    async def test_file_guards_for_send_file_and_read_image(self):
        # read_image and send_file share these guards; cover both tools in
        # both failure scenarios.
        cases = [
            (media.send_file, 'nope.txt', ('stat', 1, '', 'No such file'),
             'File not found'),
            (media.read_image, 'nope.png', ('stat', 1, '', 'No such file'),
             'File not found'),
            # Too large, per each tool's own limit.
            (media.send_file, 'big.bin',
             ('stat', 0, f'{media.MAX_SEND_FILE_BYTES + 1}\n', ''),
             'too large'),
            (media.read_image, 'big.png',
             ('stat', 0, f'{media.MAX_IMAGE_DISPLAY_BYTES + 1}\n', ''),
             'too large'),
        ]
        for tool, path, exec_response, keyword in cases:
            with self.subTest(tool=tool.__name__, path=path):
                manager = self._canned([exec_response])
                with _patched_manager(manager):
                    result = await tool(mock_ctx(bot=_FakeBot()), path)
                assert isinstance(result, str)
                self.assertIn(keyword, result)

    async def test_send_file_base64_failure(self):
        manager = self._canned([('stat', 0, '4\n', ''), ('base64', 1, '', 'Is a directory')])
        with _patched_manager(manager):
            ctx = mock_ctx(bot=_FakeBot())
            result = await media.send_file(ctx, 'dir')
        self.assertIn('Is a directory', result)


class TestReadImage(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # A fresh cache per test keeps upload records from leaking between
        # tests; restored afterwards for the other test classes.
        self._original_cache = upload_cache._recent_uploads
        upload_cache._recent_uploads = cachetools.TTLCache(maxsize=8, ttl=3600)

    def tearDown(self):
        upload_cache._recent_uploads = self._original_cache

    def _canned(self, exec_responses):
        return SandboxManager(docker=CannedDocker(exec_responses))

    async def _read_image(self, payload=b'img-bytes', path='rendered.png',
                          uploaded_by=None, broken_send_photo=False,
                          **kwargs):
        """Run read_image against a docker that serves `payload` at `path`.

        uploaded_by: record the file as a recent upload by that user, so
        the "already visible in the chat" shortcut can be exercised.
        Returns (result, bot) so callers can assert on both.
        """
        encoded = base64.b64encode(payload).decode()
        manager = self._canned([
            ('stat', 0, f'{len(payload)}\n', ''),
            ('base64', 0, encoded, ''),
        ])
        bot = _FakeBot()
        if uploaded_by is not None:
            upload_cache.record_upload(uploaded_by, f'/workspace/{path}')
        if broken_send_photo:
            async def broken(*args, **kwargs):
                raise RuntimeError('telegram down')
            bot.send_photo = broken
        with _patched_manager(manager):
            result = await media.read_image(
                mock_ctx(bot=bot), path, **kwargs)
        return result, bot

    async def test_unsupported_extension(self):
        manager = self._canned([])
        with _patched_manager(manager):
            ctx = mock_ctx(bot=_FakeBot())
            result = await media.read_image(ctx, 'file.txt')
        assert isinstance(result, str)
        self.assertIn('not a supported image type', result)

    async def test_success_sends_photo_and_returns_image(self):
        result, bot = await self._read_image(bytes(range(256)))
        assert isinstance(result, ToolCallResult)
        assert result.image is not None
        self.assertEqual(result.image.data, bytes(range(256)))
        self.assertEqual(result.image.mime, 'image/png')
        self.assertIn('/workspace/rendered.png', result.text)
        self.assertEqual(bot.sent_photos, [bytes(range(256))])

    async def test_images_not_shown_are_not_sent(self):
        # send_to_chat=False keeps a read image private.
        result, bot = await self._read_image(send_to_chat=False)
        assert isinstance(result, ToolCallResult)
        self.assertEqual(bot.sent_photos, [])
        self.assertNotIn('shown in the chat', result.text)
        self.assertIn('Image read', result.text)

    async def test_recent_upload_is_not_resent_to_the_chat(self):
        # An image the user just uploaded is already visible in the chat
        # window; read_image attaches it for the model without re-sending it.
        result, bot = await self._read_image(
            path='uploads/pic.png', uploaded_by=1)
        assert isinstance(result, ToolCallResult)
        assert result.image is not None
        self.assertEqual(result.image.data, b'img-bytes')
        self.assertEqual(bot.sent_photos, [])
        self.assertIn('recently uploaded', result.text)
        self.assertIn('attached to this result', result.text)

        # Another user's upload of the same file is not spared.
        upload_cache._recent_uploads.pop(
            (1, '/workspace/uploads/pic.png'), None)
        result, bot = await self._read_image(
            path='uploads/pic.png', uploaded_by=2)
        assert isinstance(result, ToolCallResult)
        self.assertEqual(bot.sent_photos, [b'img-bytes'])
        self.assertIn('Image shown in the chat', result.text)

    async def test_send_photo_failure_still_attaches(self):
        result, _ = await self._read_image(broken_send_photo=True)
        assert isinstance(result, ToolCallResult)
        assert result.image is not None
        self.assertEqual(result.image.data, b'img-bytes')


class TestToolResultImageContract(unittest.IsolatedAsyncioTestCase):

    def _tool_call(self, name, arguments):
        return SimpleNamespace(
            function=SimpleNamespace(name=name, arguments=arguments))

    async def test_execute_tool_call_passes_results_through(self):
        # A tool's return value reaches the model unchanged: an
        # image-bearing ToolCallResult stays one, a plain string stays a
        # string the tool message can carry verbatim.
        async def image_tool(ctx, filePath):
            return ToolCallResult(
                text="shown", image=ImageAttachment(b"img", "image/png"))

        async def text_tool(ctx):
            return "plain text result"

        expected_image = ToolCallResult(
            text="shown", image=ImageAttachment(b"img", "image/png"))
        for tool, arguments, expected in (
                (image_tool, '{"filePath": "x.png"}', expected_image),
                (text_tool, '{}', "plain text result")):
            with self.subTest(tool=tool.__name__):
                setattr(tool, "_needs_ctx", True)
                result = await run_llm_mod.execute_tool_call(
                    mock_ctx(), self._tool_call("fake_tool", arguments),
                    {"fake_tool": tool})
                self.assertEqual(result, expected)
                if isinstance(expected, ToolCallResult):
                    assert isinstance(result, ToolCallResult)
                    assert result.image is not None
                    self.assertEqual(result.image.data, b"img")

    async def test_image_tool_call_appends_user_message(self):
        payload = b'png-bytes'
        encoded = base64.b64encode(payload).decode()
        manager = SandboxManager(docker=CannedDocker([
            ('stat', 0, f'{len(payload)}\n', ''),
            ('base64', 0, encoded, ''),
        ]))
        tool_chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(
                content=None, reasoning_content=None,
                tool_calls=[SimpleNamespace(
                    index=0, id='call_1',
                    function=SimpleNamespace(
                        name='read_image',
                        arguments='{"filePath": "pic.png"}'))]),
            finish_reason='tool_calls')])
        final_chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(
                content='It is a space scene.', reasoning_content=None,
                tool_calls=None),
            finish_reason='stop')])
        responses = [tool_chunk, final_chunk]

        async def fake_create(**kwargs):
            chunk = responses.pop(0)

            async def stream():
                yield chunk
            return stream()

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=fake_create)))
        bot = _FakeBot()
        tool_defs = [{"type": "function",
                      "function": {"name": 'read_image',
                                   "parameters": {}}}]
        with _patched_manager(manager), \
             unittest.mock.patch.object(run_llm_mod, 'client', fake_client):
            ctx = mock_ctx(bot=bot, conversation=[])
            ctx.user_data.settings = SimpleNamespace(  # type: ignore[assignment]
                show_tool_output=False, model='m', temperature=1.0, top_p=0.95,
                thinking_keep=False)
            conversation = Conversation()
            result = await run_llm_mod.run_llm(
                ctx, conversation=conversation,
                tools=ToolSurface(definitions=tool_defs,
                                  functions={'read_image': media.read_image}))

        self.assertEqual(result, 'It is a space scene.')
        tool_messages = [m for m in conversation if m['role'] == 'tool']
        self.assertEqual(tool_messages[0]['content'], (
            f'Image shown in the chat: /workspace/pic.png ({len(payload)} bytes). '
            'The image is attached to this result.'))
        user_messages = [m for m in conversation if m['role'] == 'user']
        image_msg = user_messages[-1]['content']
        self.assertEqual(image_msg[0]['type'], 'text')
        self.assertEqual(image_msg[1]['type'], 'image_url')
        self.assertEqual(
            image_msg[1]['image_url']['url'],
            f'data:image/png;base64,{encoded}')


if __name__ == '__main__':
    unittest.main()
