"""Isolation guarantees of the two context levels.

A subagent runs against ``AgentContext``: no bot, no chat, no
``messaging`` handle, and its own copy of the mutable state. These tests
pin that down, since a regression here means a subagent can reach the
user's chat or the parent's coordination state.
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_inbox import SubagentInbox
from chat_update_context import (AgentContext,
                                 ChatUpdateContext)
from conversation import Conversation
from mock_context import mock_ctx

SETTINGS = SimpleNamespace(
    show_tool_output=True, model='m', temperature=1.0, top_p=0.95,
    subagent_model='sub-m')


def _parent() -> ChatUpdateContext:
    ctx = mock_ctx(settings=SETTINGS,
                   conversation=[{'role': 'system', 'content': 'sys'}])
    ctx.thread_data.main_inbox = SubagentInbox()
    ctx.thread_data.conversation_epoch = 3
    return ctx


def _child(ctx: ChatUpdateContext) -> AgentContext:
    return ctx.new_subagent_context(Conversation())


class TestContextHierarchy(unittest.TestCase):

    def test_interactive_context_is_a_chat_context(self):
        # Tools annotate the base type, so the parent's context has to be
        # accepted by anything a subagent can use.
        self.assertTrue(issubclass(ChatUpdateContext, AgentContext))

    def test_parent_has_messaging_and_child_does_not(self):
        parent = _parent()
        self.assertTrue(hasattr(parent, 'messaging'))
        self.assertFalse(hasattr(_child(parent), 'messaging'))


class TestSubagentContextIsolation(unittest.TestCase):

    def test_no_way_to_reach_telegram(self):
        child = _child(_parent())
        self.assertFalse(hasattr(child, 'bot'))
        self.assertFalse(hasattr(child, 'chat'))
        self.assertFalse(hasattr(child, 'chat_id'))
        self.assertFalse(hasattr(child, 'message'))
        self.assertFalse(hasattr(child, 'update'))
        self.assertFalse(hasattr(child, 'context'))

    def test_user_data_is_its_own_object(self):
        parent = _parent()
        child = _child(parent)
        self.assertIsNot(child.user_data, parent.user_data)
        self.assertIsNot(child.thread_data, parent.thread_data)
        child.thread_data.conversation_epoch = 99
        self.assertNotEqual(99, parent.thread_data.conversation_epoch)

    def test_parent_coordination_state_is_unreachable(self):
        child = _child(_parent())
        self.assertIsNone(child.thread_data.main_inbox)
        self.assertIsNone(child.thread_data.wakeup_task)
        self.assertFalse(child.thread_data.main_loop_active)
        self.assertEqual(child.thread_data.conversation_epoch, 0)
        self.assertIsNone(child.thread_data.generation_queue)
        self.assertIsNone(child.thread_data.generation_task)
        self.assertEqual(child.thread_data.subagents, {})

    def test_conversation_is_the_subagents_own(self):
        parent = _parent()
        own = Conversation()
        child = parent.new_subagent_context(own)
        self.assertIs(child.conversation, own)
        own.add_user('subagent thought')
        self.assertNotIn('subagent thought',
                         [m.get('content')
                          for m in parent.conversation.messages])
        child.conversation = Conversation()
        self.assertIsNot(parent.conversation, child.conversation)

    def test_settings_are_a_snapshot(self):
        parent = _parent()
        child = _child(parent)
        self.assertIsNot(child.settings, parent.settings)
        self.assertEqual(child.settings.model, parent.settings.model)
        child.settings.model = 'changed'
        self.assertEqual(parent.settings.model, 'm')

    def test_real_user_settings_are_copied(self):
        from user_settings import UserSettings
        parent = _parent()
        parent.user_data.settings = UserSettings.default()
        parent.settings.model = 'parent-model'
        child = _child(parent)
        self.assertEqual(child.settings.model, 'parent-model')
        child.settings.model = 'child-model'
        self.assertEqual(parent.settings.model, 'parent-model')

    def test_silent_surface_is_minimal(self):
        # The silent context is the tool-facing contract: identity plus the
        # agent's own state. Growing it is a deliberate decision (and a
        # review of whether the new member is truly chat-agnostic), never
        # an accident.
        public = {name for name in dir(_child(_parent()))
                  if not name.startswith('_')}
        self.assertEqual(
            public, {'user_id', 'user_data', 'thread_data',
                     'conversation', 'settings'})

    def test_sandbox_read_tracking_starts_empty(self):
        parent = _parent()
        parent.user_data.sandbox_read_files = {'/workspace/a.txt'}
        child = _child(parent)
        self.assertEqual(child.user_data.sandbox_read_files, set())


if __name__ == '__main__':
    unittest.main()
