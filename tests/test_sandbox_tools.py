import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sandbox_image', 'helpers'))

import toolbot_edit
import toolbot_glob
import toolbot_read
from chat_update_context import ChatUpdateContext
from handlers import upload_handler
from tools import sandbox_tool
from sandbox import SandboxManager


class _FakeBot:
    def __init__(self):
        self.sent = []

    async def send_document(self, *args, **kwargs):
        document = args[1] if len(args) >= 2 else kwargs.get("document")
        self.sent.append(
            {"filename": kwargs.get("filename"), "document": document})


def _mock_ctx(user_id: int = 1, bot=None) -> ChatUpdateContext:
    context = SimpleNamespace(
        bot=bot, bot_data={}, user_data={}, chat_data={})
    message = SimpleNamespace(
        text='', caption=None,
        chat=SimpleNamespace(id=user_id, type='private'),
        from_user=SimpleNamespace(id=user_id, first_name='T', last_name='U'))
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


class CannedDocker:
    """Fake docker CLI with scripted exec responses.

    exec responses are matched in order against the command string; the
    first matching (substring, returncode, stdout, stderr) tuple is used.
    inspect/run/rm/ps behave like the FakeDocker in test_sandbox.
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


def _run_main(argv: list[str], fn) -> dict:
    """Run a helper's main() with the given argv, capture its JSON stdout."""
    original = sys.argv
    sys.argv = ['helper'] + argv
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            fn()
    finally:
        sys.argv = original
    return json.loads(captured.getvalue())


class TestEditHelper(unittest.TestCase):

    def _replace(self, content, old, new, replace_all=False):
        return toolbot_edit.replace(content, old, new, replace_all)

    def test_exact_replacement(self):
        self.assertEqual(self._replace('foo\nbar\nbaz\n', 'bar', 'qux'),
                         'foo\nqux\nbaz\n')

    def test_not_found(self):
        with self.assertRaises(ValueError, msg='Could not find oldString'):
            self._replace('foo\n', 'nope', 'x')

    def test_multiple_matches_error(self):
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

    def test_identical_strings_error(self):
        with self.assertRaises(ValueError, msg='identical'):
            self._replace('x\n', 'x', 'x')

    def test_empty_oldstring_error(self):
        with self.assertRaises(ValueError, msg='oldString cannot be empty'):
            self._replace('x\n', '', 'y')

    def test_line_trimmed_fallback(self):
        content = 'def f():\n\treturn 1\n'
        old = 'def f():\n    return 1'
        result = self._replace(content, old, 'def f():\n\treturn 2')
        self.assertEqual(result, 'def f():\n\treturn 2\n')

    def test_indentation_flexible_fallback(self):
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
            original_stdin = sys.stdin
            sys.stdin = io.StringIO(json.dumps(payload))
            try:
                result = _run_main([], toolbot_edit.main)
            finally:
                sys.stdin = original_stdin
            self.assertTrue(result['ok'])
            with open(path) as f:
                self.assertEqual(f.read(), 'foo\nqux\n')

    def test_helper_script_not_found(self):
        payload = {'path': 'missing.txt', 'old_string': 'a', 'new_string': 'b'}
        original_stdin = sys.stdin
        sys.stdin = io.StringIO(json.dumps(payload))
        try:
            result = _run_main([], toolbot_edit.main)
        finally:
            sys.stdin = original_stdin
        self.assertFalse(result['ok'])


class TestReadHelper(unittest.TestCase):

    def _read(self, *args) -> dict:
        return _run_main(list(args), toolbot_read.main)

    def test_line_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.txt')
            with open(path, 'w') as f:
                f.write('a\nb\nc\n')
            result = self._read('--path', path)
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
            result = self._read('--path', path, '--offset', '5', '--limit', '3')
            data = result['data']
            self.assertEqual([line['n'] for line in data['lines']], [5, 6, 7])
            self.assertTrue(data['more'])

    def test_offset_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.txt')
            with open(path, 'w') as f:
                f.write('a\n')
            result = self._read('--path', path, '--offset', '10')
            self.assertFalse(result['ok'])
            self.assertIn('out of range', result['error'])

    def test_long_line_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.txt')
            with open(path, 'w') as f:
                f.write('x' * 3000 + '\n')
            result = self._read('--path', path)
            text = result['data']['lines'][0]['text']
            self.assertEqual(
                len(text), 2000 + len(toolbot_read.MAX_LINE_SUFFIX))
            self.assertTrue(text.endswith('chars)'))

    def test_directory_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'sub'))
            with open(os.path.join(tmp, 'file.txt'), 'w') as f:
                f.write('x')
            result = self._read('--path', tmp)
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
            result = self._read('--path', path)
            self.assertFalse(result['ok'])
            self.assertIn('binary', result['error'])

    def test_not_found_with_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'app.py'), 'w') as f:
                f.write('x')
            result = self._read('--path', os.path.join(tmp, 'app'))
            self.assertFalse(result['ok'])
            self.assertIn('Did you mean', result['error'])
            self.assertIn('app.py', result['error'])

    def test_not_found_no_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._read('--path', os.path.join(tmp, 'missing.txt'))
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
            result = _run_main(
                ['--pattern', '*', '--path', file_path], toolbot_glob.main)
            self.assertFalse(result['ok'])
            self.assertIn('directory', result['error'])


class TestUploadNaming(unittest.TestCase):

    def test_no_collision(self):
        self.assertEqual(upload_handler._unique_name("f.txt", set()), "f.txt")

    def test_collision_gets_suffix(self):
        self.assertEqual(
            upload_handler._unique_name("f.txt", {"f.txt"}), "f-1.txt")

    def test_multi_collision_increments(self):
        self.assertEqual(
            upload_handler._unique_name(
                "f.txt", {"f.txt", "f-1.txt", "f-2.txt"}),
            "f-3.txt")

    def test_suffix_keeps_extension(self):
        self.assertEqual(
            upload_handler._unique_name("photo.jpg", {"photo.jpg"}),
            "photo-1.jpg")

    def test_no_extension(self):
        self.assertEqual(
            upload_handler._unique_name("README", {"README"}), "README-1")

    def test_fallback_name_from_kind_and_path(self):
        class Document:
            pass
        file = SimpleNamespace(file_path="voice/file_123.oga")
        name = upload_handler._fallback_name(Document(), file)
        self.assertTrue(name.startswith("document_"))
        self.assertTrue(name.endswith(".oga"))

    def test_fallback_name_photo_kind(self):
        class PhotoSize:
            pass
        file = SimpleNamespace(file_path="photos/file_5.jpg")
        name = upload_handler._fallback_name(PhotoSize(), file)
        self.assertTrue(name.startswith("photo_"))
        self.assertTrue(name.endswith(".jpg"))

    def test_fallback_name_missing_path(self):
        class Voice:
            pass
        file = SimpleNamespace(file_path=None)
        name = upload_handler._fallback_name(Voice(), file)
        self.assertRegex(name, r"^voice_\d+$")


class TestSandboxToolHelpers(unittest.TestCase):

    def test_resolve_path_relative(self):
        self.assertEqual(sandbox_tool._resolve_path('src/app.py'),
                         '/workspace/src/app.py')

    def test_resolve_path_absolute(self):
        self.assertEqual(sandbox_tool._resolve_path('/etc/hosts'), '/etc/hosts')

    def test_resolve_path_normalizes(self):
        self.assertEqual(sandbox_tool._resolve_path('src/../app.py'),
                         '/workspace/app.py')

    def test_parse_rg_json(self):
        events = '\n'.join([
            json.dumps({'type': 'begin', 'data': {'path': {'text': 'a.py'}}}),
            json.dumps({'type': 'match', 'data': {
                'path': {'text': 'a.py'}, 'line_number': 3,
                'lines': {'text': 'needle here\n'}}}),
            json.dumps({'type': 'match', 'data': {
                'path': {'text': 'a.py'}, 'line_number': 7,
                'lines': {'text': 'another needle\n'}}}),
            json.dumps({'type': 'end', 'data': {}}),
            'partial-truncated-garbage',
        ])
        matches = sandbox_tool._parse_rg_json(events)
        self.assertEqual(matches, [
            {'path': 'a.py', 'line': 3, 'text': 'needle here'},
            {'path': 'a.py', 'line': 7, 'text': 'another needle'},
        ])

    def test_parse_rg_json_garbage_only(self):
        self.assertEqual(sandbox_tool._parse_rg_json('not json\n'), [])


class TestSandboxTools(unittest.IsolatedAsyncioTestCase):

    def _canned(self, exec_responses):
        return SandboxManager(docker=CannedDocker(exec_responses))

    async def test_bash_formats_output(self):
        manager = self._canned([('echo', 0, 'hello\n', '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.bash(ctx, 'echo hello')
        self.assertIn('Return code: 0', result)
        self.assertIn('stdout:\nhello', result)

    async def test_write_creates_and_marks_read(self):
        manager = self._canned([('test -e', 1, '', ''), ('mkdir', 0, '', '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.write(ctx, 'src/new.txt', 'data')
        self.assertEqual(result, 'Wrote file successfully.')
        self.assertIn('/workspace/src/new.txt', ctx.user_data['sandbox_read_files'])

    async def test_write_requires_read_first(self):
        manager = self._canned([('test -e', 0, '', '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.write(ctx, 'existing.txt', 'data')
        self.assertIn('has not been read', result)

    async def test_edit_requires_read_first(self):
        manager = self._canned([])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.edit(ctx, 'f.txt', 'old', 'new')
        self.assertIn('has not been read', result)

    async def test_edit_runs_helper(self):
        helper_result = json.dumps(
            {'ok': True, 'data': {'output': 'Edit applied successfully.'}})
        manager = self._canned([('toolbot_edit.py', 0, helper_result, '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            ctx.user_data['sandbox_read_files'] = {'/workspace/f.txt'}
            result = await sandbox_tool.edit(ctx, 'f.txt', 'old', 'new')
        self.assertEqual(result, 'Edit applied successfully.')

    async def test_edit_identical_strings(self):
        manager = self._canned([])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            ctx.user_data['sandbox_read_files'] = {'/workspace/f.txt'}
            result = await sandbox_tool.edit(ctx, 'f.txt', 'same', 'same')
        self.assertIn('identical', result)

    async def test_read_formats_lines(self):
        helper_result = json.dumps({'ok': True, 'data': {
            'type': 'file', 'path': '/workspace/f.txt',
            'lines': [{'n': 1, 'text': 'alpha'}, {'n': 2, 'text': 'beta'}],
            'lineStart': 1, 'lineEnd': 2, 'totalLines': 2,
            'more': False, 'cut': False}})
        manager = self._canned([('toolbot_read.py', 0, helper_result, '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.read(ctx, 'f.txt')
        self.assertIn('1: alpha', result)
        self.assertIn('2: beta', result)
        self.assertIn('End of file - total 2 lines', result)
        self.assertIn('/workspace/f.txt', ctx.user_data['sandbox_read_files'])

    async def test_read_error(self):
        helper_result = json.dumps(
            {'ok': False, 'error': 'File not found: /workspace/x'})
        manager = self._canned([('toolbot_read.py', 0, helper_result, '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.read(ctx, 'x')
        self.assertIn('File not found', result)

    async def test_read_directory(self):
        helper_result = json.dumps({'ok': True, 'data': {
            'type': 'directory', 'path': '/workspace',
            'entries': ['src/', 'app.py'], 'offset': 1,
            'totalEntries': 2, 'truncated': False}})
        manager = self._canned([('toolbot_read.py', 0, helper_result, '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.read(ctx, '.')
        self.assertIn('<type>directory</type>', result)
        self.assertIn('src/', result)
        self.assertNotIn('/workspace',
                         ctx.user_data.get('sandbox_read_files', set()))

    async def test_grep_formats_matches(self):
        events = '\n'.join([
            json.dumps({'type': 'match', 'data': {
                'path': {'text': 'src/a.py'}, 'line_number': 4,
                'lines': {'text': 'find_me()\n'}}}),
            json.dumps({'type': 'match', 'data': {
                'path': {'text': 'src/b.py'}, 'line_number': 9,
                'lines': {'text': 'find_me()\n'}}}),
        ])
        manager = self._canned([('rg', 0, events, '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.grep(ctx, 'find_me')
        self.assertIn('Found 2 matches', result)
        self.assertIn('/workspace/src/a.py:', result)
        self.assertIn('  Line 4: find_me()', result)
        self.assertIn('  Line 9: find_me()', result)

    async def test_grep_no_matches(self):
        manager = self._canned([('rg', 0, '', '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.grep(ctx, 'nothing')
        self.assertEqual(result, 'No files found')

    async def test_grep_reports_rg_error(self):
        manager = self._canned([('rg', 0, '', 'regex parse error')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.grep(ctx, '[')
        self.assertIn('regex parse error', result)

    async def test_grep_truncates(self):
        events = '\n'.join(
            json.dumps({'type': 'match', 'data': {
                'path': {'text': 'a.py'}, 'line_number': i,
                'lines': {'text': 'hit\n'}}})
            for i in range(sandbox_tool.MAX_GREP_MATCHES + 10))
        manager = self._canned([('rg', 0, events, '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.grep(ctx, 'hit')
        self.assertIn('Found 100 matches (more matches available)', result)
        self.assertIn('Results truncated', result)

    async def test_glob_formats_files(self):
        helper_result = json.dumps({'ok': True, 'data': {
            'files': ['/workspace/a.py', '/workspace/b.py'],
            'truncated': False}})
        manager = self._canned([('toolbot_glob.py', 0, helper_result, '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.glob(ctx, '*.py')
        self.assertIn('/workspace/a.py', result)
        self.assertIn('/workspace/b.py', result)

    async def test_glob_no_files(self):
        helper_result = json.dumps(
            {'ok': True, 'data': {'files': [], 'truncated': False}})
        manager = self._canned([('toolbot_glob.py', 0, helper_result, '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.glob(ctx, '*.nope')
        self.assertEqual(result, 'No files found')

    async def test_apply_patch_success(self):
        summary = 'Success. Updated the following files:\nA src/new.py\nM app.py'
        manager = self._canned([('apply-patch-py', 0, summary + '\n', '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.apply_patch(
                ctx, '*** Begin Patch\n*** End Patch')
        self.assertIn('A src/new.py', result)
        self.assertIn('/workspace/src/new.py', ctx.user_data['sandbox_read_files'])

    async def test_apply_patch_error(self):
        manager = self._canned(
            [('apply-patch-py', 1, '', 'patch rejected: bad header')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx()
            result = await sandbox_tool.apply_patch(ctx, 'garbage')
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
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx(bot=bot)
            result = await sandbox_tool.send_file(ctx, 'f.txt')
        self.assertEqual(result, 'Sent f.txt (11 bytes).')
        self.assertEqual(bot.sent[0]['filename'], 'f.txt')
        self.assertEqual(bot.sent[0]['document'], b'hello world')

    async def test_send_file_not_found(self):
        manager = self._canned([('stat', 1, '', 'No such file')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx(bot=_FakeBot())
            result = await sandbox_tool.send_file(ctx, 'nope.txt')
        self.assertIn('File not found', result)

    async def test_send_file_too_large(self):
        manager = self._canned(
            [('stat', 0, f'{sandbox_tool.MAX_SEND_FILE_BYTES + 1}\n', '')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx(bot=_FakeBot())
            result = await sandbox_tool.send_file(ctx, 'big.bin')
        self.assertIn('too large', result)

    async def test_send_file_binary_content_preserved(self):
        payload = bytes(range(256))
        encoded = base64.b64encode(payload).decode()
        manager = self._canned([
            ('stat', 0, f'{len(payload)}\n', ''),
            ('base64', 0, encoded, ''),
        ])
        bot = _FakeBot()
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx(bot=bot)
            await sandbox_tool.send_file(ctx, 'blob.bin')
        self.assertEqual(bot.sent[0]['document'], payload)

    async def test_send_file_base64_failure(self):
        manager = self._canned([('stat', 0, '4\n', ''), ('base64', 1, '', 'Is a directory')])
        with unittest.mock.patch.object(sandbox_tool, 'manager', manager):
            ctx = _mock_ctx(bot=_FakeBot())
            result = await sandbox_tool.send_file(ctx, 'dir')
        self.assertIn('Is a directory', result)


if __name__ == '__main__':
    unittest.main()
