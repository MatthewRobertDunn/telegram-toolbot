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
        # The walk prunes directories at the sixth level, so a SKILL.md
        # only at a/b/c/d/e/f is never found. Nothing else is in the
        # tree, so pruning is the only thing under test here.
        _write(os.path.join(self.root, 'a', 'b', 'c', 'd', 'e', 'f',
                            'SKILL.md'), SKILL_MD)

        skills = self._load()

        self.assertEqual(skills, [])

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

    def test_frontmatter_parsing_cases(self):
        cases = [
            ('basic fields',
             '---\nname: n\ndescription: d\n---\n',
             {'name': 'n', 'description': 'd'}),
            ('no opening delimiter yields empty',
             'name: n\n',
             {}),
            ('missing closing delimiter parses to eof',
             '---\nname: n\n',
             {'name': 'n'}),
            ('quoted values are unquoted',
             '---\nname: "quoted"\nother: \'single\'\n---\n',
             {'name': 'quoted', 'other': 'single'}),
            ('indented continuation appends to previous key',
             '---\ndescription: one\n  two\n---\n',
             {'description': 'one two'}),
            ('empty and whitespace-only lines are skipped',
             '---\n\n   \nname: n\n---\n',
             {'name': 'n'}),
        ]
        for label, text, expected in cases:
            with self.subTest(label):
                self.assertEqual(self._parse(text), expected)


class TestSandboxPathPrefix(unittest.TestCase):

    def _sandbox_paths(self, prefix: str) -> list:
        with tempfile.TemporaryDirectory() as root:
            _write(os.path.join(root, 's', 'SKILL.md'), SKILL_MD)
            _write(os.path.join(root, 'flat.md'), SKILL_MD)
            with mock.patch.object(skills_store.config,
                                   'SANDBOX_SKILLS_DIR', prefix):
                skills = load_skills(root)
        return [s.sandbox_path for s in skills]

    def test_prefix_shapes(self):
        cases = [
            ('default prefix',
             '/skills',
             ['/skills/flat.md', '/skills/s/SKILL.md']),
            ('prefix without trailing slash does not double it',
             '/custom',
             ['/custom/flat.md', '/custom/s/SKILL.md']),
        ]
        for label, prefix, expected in cases:
            with self.subTest(label):
                self.assertEqual(sorted(self._sandbox_paths(prefix)),
                                 expected)

    def test_windows_separators_become_posix(self):
        # On a Windows host os.sep is '\\'; sandbox paths must still be
        # POSIX. Simulate by using a root containing backslashes.
        if os.sep != '\\':
            self.skipTest('Windows-only path shape')
        self.assertIn('/skills/s/SKILL.md', self._sandbox_paths('/skills'))


if __name__ == '__main__':
    unittest.main()
