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
                                 ChatUpdateContext,
                                 create_chat_update_context)
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


def _ptb_pair(store: dict, thread_id) -> ChatUpdateContext:
    """Build a context through the real factory, as PTB would per update."""
    message = SimpleNamespace(text='hi', caption=None,
                              message_thread_id=thread_id)
    chat = SimpleNamespace(id=7, type='private')
    user = SimpleNamespace(id=1, first_name='T', last_name='U')
    update = SimpleNamespace(effective_chat=chat, effective_user=user,
                             message=message, effective_message=message)
    context = SimpleNamespace(user_data=store, bot=None, bot_data={})
    return create_chat_update_context(update, context)  # type: ignore[arg-type]


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

    def test_subagent_state_is_independent(self):
        parent = _parent()
        own = Conversation()
        child = parent.new_subagent_context(own)
        self.assertIsNot(child.user_data, parent.user_data)
        self.assertIsNot(child.thread_data, parent.thread_data)
        child.thread_data.conversation_epoch = 99
        self.assertNotEqual(99, parent.thread_data.conversation_epoch)
        # The conversation is the subagent's own object: a message added
        # to it does not appear in the parent's history.
        self.assertIs(child.conversation, own)
        own.add_user('subagent thought')
        self.assertNotIn('subagent thought',
                         [m.get('content')
                          for m in parent.conversation.messages])
        child2 = _child(parent)
        self.assertIsNot(child2.conversation, own)

    def test_settings_are_a_snapshot(self):
        parent = _parent()
        child = _child(parent)
        self.assertIsNot(child.settings, parent.settings)
        self.assertEqual(child.settings.model, parent.settings.model)
        child.settings.model = 'changed'
        self.assertEqual(parent.settings.model, 'm')

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


class TestThreadKeyedStorage(unittest.TestCase):

    def test_same_thread_returns_shared_state(self):
        store: dict = {}
        first = _ptb_pair(store, None)
        second = _ptb_pair(store, None)
        self.assertIs(first.thread_data, second.thread_data)
        self.assertIs(first.user_data, second.user_data)

    def test_different_threads_are_isolated_but_share_the_user(self):
        store: dict = {}
        plain = _ptb_pair(store, None)
        topic = _ptb_pair(store, 2)
        self.assertIsNone(plain.message_thread_id)
        self.assertEqual(topic.message_thread_id, 2)
        self.assertIsNot(plain.thread_data, topic.thread_data)
        self.assertIs(plain.user_data, topic.user_data)
        plain.thread_data.conversation = Conversation()
        self.assertIsNone(topic.thread_data.conversation)


if __name__ == '__main__':
    unittest.main()
