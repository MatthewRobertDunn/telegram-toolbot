"""Tests for opencode.ai-compatible skill discovery."""

import os
import tempfile
import unittest
from unittest import mock

import skills_store
from skills_store import SkillInfo, get_skill, load_skills


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


SKILL_MD = """---
name: Display Name
description: Does things well.
---

# Body
"""


class TestSkillInfo(unittest.TestCase):

    def test_equality_compares_all_fields(self):
        a = SkillInfo('x', 'd', '/skills/x.md')
        b = SkillInfo('x', 'd', '/skills/x.md')
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, SkillInfo('x', 'other', '/skills/x.md'))
        self.assertNotEqual(a, 'not a skill')

    def test_frozen(self):
        skill = SkillInfo('x', 'd', '/skills/x.md')
        with self.assertRaises(Exception):
            skill.name = 'y'  # type: ignore[misc]

    def test_positional_construction_and_repr(self):
        skill = SkillInfo('x', 'd', '/skills/x.md')
        self.assertEqual(
            repr(skill),
            "SkillInfo(name='x', description='d', sandbox_path='/skills/x.md')")


class TestSkillDiscovery(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.addCleanup(lambda: load_skills.__globals__.update(
            SKILLS=[]))

    def _load(self):
        return load_skills(self.root)

    def test_discovers_directory_flat_and_nested_forms(self):
        _write(os.path.join(self.root, 'flat.md'), SKILL_MD)
        _write(os.path.join(self.root, 'git-release', 'SKILL.md'), SKILL_MD)
        _write(os.path.join(self.root, 'teams', 'release2', 'SKILL.md'),
               SKILL_MD)

        skills = self._load()

        by_name = {s.name: s for s in skills}
        self.assertEqual(
            set(by_name), {'flat', 'git-release', 'release2'})
        for skill in skills:
            self.assertEqual(skill.description, 'Does things well.')
        self.assertEqual(by_name['flat'].sandbox_path, '/skills/flat.md')
        self.assertEqual(by_name['git-release'].sandbox_path,
                         '/skills/git-release/SKILL.md')
        self.assertEqual(by_name['release2'].sandbox_path,
                         '/skills/teams/release2/SKILL.md')

    def test_id_comes_from_path_not_frontmatter_name(self):
        _write(os.path.join(self.root, 'my-dir', 'SKILL.md'), SKILL_MD)

        skills = self._load()

        self.assertEqual([s.name for s in skills], ['my-dir'])

    def test_missing_description_defaults_to_empty(self):
        _write(os.path.join(self.root, 'nodesc', 'SKILL.md'),
               '---\nname: X\n---\n\nBody\n')

        skills = self._load()

        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].description, '')

    def test_duplicate_ids_last_wins(self):
        # The flat file is yielded first, the directory form later, so
        # the directory definition overrides it for the shared ID.
        _write(os.path.join(self.root, 'dup.md'),
               '---\ndescription: flat\n---\n')
        _write(os.path.join(self.root, 'nested', 'dup', 'SKILL.md'),
               '---\ndescription: nested\n---\n')

        skills = self._load()

        dupes = [s for s in skills if s.name == 'dup']
        self.assertEqual(len(dupes), 1)
        self.assertEqual(dupes[0].description, 'nested')
        self.assertEqual(dupes[0].sandbox_path, '/skills/nested/dup/SKILL.md')

    def test_skips_hidden_and_node_modules(self):
        _write(os.path.join(self.root, '.hidden', 'SKILL.md'), SKILL_MD)
        _write(os.path.join(self.root, 'node_modules', 'dep', 'SKILL.md'),
               SKILL_MD)
        _write(os.path.join(self.root, '.dot.md'), SKILL_MD)
        _write(os.path.join(self.root, 'visible', 'SKILL.md'), SKILL_MD)

        skills = self._load()

        self.assertEqual([s.name for s in skills], ['visible'])

    def test_depth_cutoff(self):
        # The walk does not descend past the sixth directory level:
        # SKILL.md five levels below the root is found, six is not.
        deep_five = os.path.join(self.root, 'a', 'b', 'c', 'd')
        deep_six = os.path.join(deep_five, 'e')
        _write(os.path.join(deep_five, 'SKILL.md'), SKILL_MD)
        _write(os.path.join(deep_six, 'SKILL.md'), SKILL_MD)

        skills = self._load()

        self.assertEqual([s.name for s in skills], ['d'])

    def test_skill_dir_is_not_searched_deeper(self):
        # Reference material inside a skill directory is not offered
        # as its own skill.
        _write(os.path.join(self.root, 's', 'SKILL.md'), SKILL_MD)
        _write(os.path.join(self.root, 's', 'notes.md'), SKILL_MD)

        skills = self._load()

        self.assertEqual([s.name for s in skills], ['s'])

    def test_malformed_frontmatter_is_not_fatal(self):
        # No frontmatter block at all: discovered, empty description.
        _write(os.path.join(self.root, 'plain.md'), '# just a body\n')
        # A malformed line inside frontmatter is skipped, the rest kept.
        _write(os.path.join(self.root, 'odd', 'SKILL.md'),
               '---\nthis is not a pair\ndescription: kept\n---\n')

        skills = self._load()

        by_name = {s.name: s for s in skills}
        self.assertEqual(by_name['plain'].description, '')
        self.assertEqual(by_name['odd'].description, 'kept')

    def test_missing_dir_yields_empty_list(self):
        skills = load_skills(os.path.join(self.root, 'nope'))

        self.assertEqual(skills, [])
        self.assertEqual(skills_store.SKILLS, [])

    def test_unreadable_file_is_skipped(self):
        _write(os.path.join(self.root, 'good', 'SKILL.md'), SKILL_MD)
        bad = os.path.join(self.root, 'bad.md')
        _write(bad, SKILL_MD)
        os.chmod(bad, 0o000)
        self.addCleanup(os.chmod, bad, 0o644)
        try:
            with open(bad, encoding='utf-8'):
                pass
            self.skipTest('file permissions not enforceable here')
        except OSError:
            pass

        skills = self._load()
        self.assertEqual([s.name for s in skills], ['good'])

    def test_get_skill(self):
        _write(os.path.join(self.root, 'alpha', 'SKILL.md'), SKILL_MD)
        self._load()

        found = get_skill('alpha')
        self.assertIsInstance(found, SkillInfo)
        assert found is not None
        self.assertEqual(found.name, 'alpha')
        self.assertIsNone(get_skill('missing'))

    def test_get_active_skills_returns_a_copy(self):
        _write(os.path.join(self.root, 'a.md'), SKILL_MD)
        self._load()

        active = skills_store.get_active_skills()
        self.assertEqual([s.name for s in active], ['a'])
        active.clear()
        self.assertEqual([s.name for s in skills_store.get_active_skills()],
                         ['a'])


class TestFrontmatterParsing(unittest.TestCase):

    def _parse(self, text: str) -> dict:
        return skills_store._parse_frontmatter(text)

    def test_basic_fields(self):
        self.assertEqual(self._parse('---\nname: n\ndescription: d\n---\n'),
                         {'name': 'n', 'description': 'd'})

    def test_no_opening_delimiter_yields_empty(self):
        self.assertEqual(self._parse('name: n\n'), {})

    def test_missing_closing_delimiter_parses_to_eof(self):
        self.assertEqual(self._parse('---\nname: n\n'), {'name': 'n'})

    def test_quoted_values_are_unquoted(self):
        self.assertEqual(
            self._parse('---\nname: "quoted"\nother: \'single\'\n---\n'),
            {'name': 'quoted', 'other': 'single'})

    def test_indented_continuation_appends_to_previous_key(self):
        self.assertEqual(
            self._parse('---\ndescription: one\n  two\n---\n'),
            {'description': 'one two'})

    def test_empty_lines_and_whitespace_only_lines_are_skipped(self):
        self.assertEqual(self._parse('---\n\n   \nname: n\n---\n'),
                         {'name': 'n'})


class TestSandboxPathPrefix(unittest.TestCase):

    def _sandbox_paths(self, prefix: str) -> list:
        with tempfile.TemporaryDirectory() as root:
            _write(os.path.join(root, 's', 'SKILL.md'), SKILL_MD)
            _write(os.path.join(root, 'flat.md'), SKILL_MD)
            with mock.patch.object(skills_store.config,
                                   'SANDBOX_SKILLS_DIR', prefix):
                skills = load_skills(root)
        return [s.sandbox_path for s in skills]

    def test_default_prefix(self):
        self.assertEqual(
            sorted(self._sandbox_paths('/skills')),
            ['/skills/flat.md', '/skills/s/SKILL.md'])

    def test_prefix_without_trailing_slash_does_not_double_it(self):
        self.assertIn('/custom/s/SKILL.md', self._sandbox_paths('/custom'))

    def test_windows_separators_become_posix(self):
        # On a Windows host os.sep is '\\'; sandbox paths must still be
        # POSIX. Simulate by using a root containing backslashes.
        if os.sep != '\\':
            self.skipTest('Windows-only path shape')
        self.assertIn('/skills/s/SKILL.md', self._sandbox_paths('/skills'))


if __name__ == '__main__':
    unittest.main()
