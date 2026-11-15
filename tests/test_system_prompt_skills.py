import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import skills_store
from mock_context import mock_ctx
from skills_store import SkillInfo


def _settings():
    return SimpleNamespace(system_prompt='base prompt', user_profile=None)


class TestSystemPromptSkills(unittest.TestCase):

    def test_lists_each_active_skill(self):
        skills = [
            SkillInfo('alpha', 'Does A.', '/skills/alpha/SKILL.md'),
            SkillInfo('beta', 'Does B.', '/skills/beta.md'),
        ]
        ctx = mock_ctx(settings=_settings())
        with mock.patch.object(skills_store, 'SKILLS', skills):
            prompt = ctx.system_prompt

        self.assertIn('# Available skills', prompt)
        self.assertIn('## alpha\n- Path: /skills/alpha/SKILL.md\nDoes A.',
                      prompt)
        self.assertIn('## beta\n- Path: /skills/beta.md\nDoes B.', prompt)
        self.assertTrue(prompt.startswith('base prompt'))

    def test_omits_section_when_no_active_skills(self):
        ctx = mock_ctx(settings=_settings())
        with mock.patch.object(skills_store, 'SKILLS', []):
            self.assertEqual(ctx.system_prompt, 'base prompt')


if __name__ == '__main__':
    unittest.main()
