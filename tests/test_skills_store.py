import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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


class TestSkillsStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.addCleanup(lambda: load_skills.__globals__.update(
            SKILLS=[]))

    def test_discovers_directory_flat_and_nested_forms(self):
        _write(os.path.join(self.root, 'flat.md'), SKILL_MD)
        _write(os.path.join(self.root, 'git-release', 'SKILL.md'), SKILL_MD)
        _write(os.path.join(self.root, 'teams', 'release2', 'SKILL.md'),
               SKILL_MD)

        skills = load_skills(self.root)

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

        skills = load_skills(self.root)

        self.assertEqual([s.name for s in skills], ['my-dir'])

    def test_missing_description_defaults_to_empty(self):
        _write(os.path.join(self.root, 'nodesc', 'SKILL.md'),
               '---\nname: X\n---\n\nBody\n')

        skills = load_skills(self.root)

        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].description, '')

    def test_skips_hidden_and_node_modules(self):
        _write(os.path.join(self.root, '.hidden', 'SKILL.md'), SKILL_MD)
        _write(os.path.join(self.root, 'node_modules', 'dep', 'SKILL.md'),
               SKILL_MD)
        _write(os.path.join(self.root, '.dot.md'), SKILL_MD)
        _write(os.path.join(self.root, 'visible', 'SKILL.md'), SKILL_MD)

        skills = load_skills(self.root)

        self.assertEqual([s.name for s in skills], ['visible'])

    def test_missing_dir_yields_empty_list(self):
        skills = load_skills(os.path.join(self.root, 'nope'))

        self.assertEqual(skills, [])
        self.assertEqual(skills_store.SKILLS, [])

    def test_get_skill(self):
        _write(os.path.join(self.root, 'alpha', 'SKILL.md'), SKILL_MD)
        load_skills(self.root)

        found = get_skill('alpha')
        self.assertIsInstance(found, SkillInfo)
        assert found is not None
        self.assertEqual(found.name, 'alpha')
        self.assertIsNone(get_skill('missing'))


if __name__ == '__main__':
    unittest.main()
