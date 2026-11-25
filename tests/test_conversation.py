import base64
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation import (Conversation, ConversationMessage,
                          ConversationMessageType, CUSTOM, IMAGE_PLACEHOLDER,
                          MAX_IMAGES, RegenerationAction,
                          estimate_message_tokens, is_thinking_message)


class TestMessageNormalization(unittest.TestCase):
    """The constructor takes plain dicts freely and wraps them, so every
    stored message is a ConversationMessage instance."""

    def test_plain_dicts_are_wrapped_at_every_entry_point(self):
        raw = {'role': 'user', 'content': 'hi'}
        conversations = [Conversation([raw])]
        via_set = Conversation()
        via_set.set_messages([raw])
        conversations.append(via_set)
        for label, conversation in zip(('constructor', 'set_messages'),
                                       conversations):
            with self.subTest(entry_point=label):
                self.assertEqual(len(conversation.messages), 1)
                self.assertIsInstance(conversation.messages[0],
                                      ConversationMessage)
                self.assertEqual(conversation.messages[0], raw)

    def test_wrapping_keeps_your_messages(self):
        message = ConversationMessage({'role': 'user', 'content': 'hi'})
        conversation = Conversation([message])
        self.assertIs(conversation.messages[0], message)


def _user(text):
    return {'role': 'user', 'content': text}


class TestConversationIdentity(unittest.TestCase):

    def test_id_survives_mutations(self):
        conversation = Conversation()
        original_id = conversation.id
        conversation.add_user('hello')
        conversation.add_assistant('hi')
        conversation.set_system_head('prompt')
        self.assertEqual(conversation.id, original_id)


class TestTelegramIds(unittest.TestCase):

    def test_add_helpers_store_telegram_message_ids_in_custom(self):
        conversation = Conversation()
        user = conversation.add_user('hello', telegram_message_ids=[42])
        # A single id is stored as a list, just like a list of them.
        assistant = conversation.add_assistant('hi', telegram_message_ids=99)
        self.assertEqual(user['custom'], {'telegram_message_ids': [42]})
        self.assertEqual(assistant['custom'], {'telegram_message_ids': [99]})

    def test_no_telegram_message_id_leaves_no_custom_key(self):
        conversation = Conversation()
        self.assertNotIn('custom', conversation.add_user('hello'))
        message = conversation.add_assistant('hi')
        message.add_telegram_message_id(None)
        self.assertNotIn('custom', message)

    def test_add_telegram_message_id_collects_every_chunk(self):
        conversation = Conversation()
        message = conversation.add_assistant('a long reply')
        message.add_telegram_message_id(7)
        message.add_telegram_message_id(8)
        self.assertEqual(message['custom'],
                         {'telegram_message_ids': [7, 8]})

    def test_to_openai_strips_custom_metadata(self):
        conversation = Conversation()
        conversation.add_user('hello', telegram_message_ids=[42])
        conversation.add_assistant('hi', telegram_message_ids=[99])
        self.assertEqual(conversation.to_openai(),
                         [{'role': 'user', 'content': 'hello'},
                          {'role': 'assistant', 'content': 'hi'}])
        # The stored messages keep their metadata.
        self.assertEqual(conversation.messages[0]['custom'],
                         {'telegram_message_ids': [42]})


class TestConversationHelpers(unittest.TestCase):

    def test_add_returns_the_message(self):
        conversation = Conversation()
        message = conversation.add('user', 'hi')
        self.assertEqual(conversation.messages, [message])
        self.assertEqual(message, {'role': 'user', 'content': 'hi'})

    def test_add_tool_carries_the_tool_call_id(self):
        conversation = Conversation()
        message = conversation.add_tool('result text', 'call-1')
        self.assertEqual(message,
                         {'role': 'tool', 'content': 'result text',
                          'tool_call_id': 'call-1'})
        self.assertEqual(conversation.messages, [message])

    def test_set_system_head_creates_head_when_missing(self):
        conversation = Conversation()
        conversation.add_user('hi')
        conversation.set_system_head('prompt')
        self.assertEqual(conversation.messages,
                         [{'role': 'system', 'content': 'prompt'},
                          {'role': 'user', 'content': 'hi'}])
        # Calling it again replaces the head in place rather than stacking.
        conversation.set_system_head('newer')
        self.assertEqual(conversation.messages,
                         [{'role': 'system', 'content': 'newer'},
                          {'role': 'user', 'content': 'hi'}])

    def test_head_is_system(self):
        conversation = Conversation()
        self.assertFalse(conversation.head_is_system)
        conversation.add_user('hi')
        self.assertFalse(conversation.head_is_system)
        conversation.set_system_head('prompt')
        self.assertTrue(conversation.head_is_system)


class TestMessageType(unittest.TestCase):

    def test_roundtrip_sets_the_tag(self):
        message = ConversationMessage({'role': 'user', 'content': 'hi'})
        message.message_type = ConversationMessageType.IMPERSONATION
        self.assertEqual(message.message_type,
                         ConversationMessageType.IMPERSONATION)
        self.assertEqual(message[CUSTOM], {'message_type': 'impersonation'})

    def test_setting_none_clears_the_tag(self):
        message = ConversationMessage({
            'role': 'user', 'content': 'hi', CUSTOM: {'message_type': 'imp'}})
        message.message_type = None
        self.assertIsNone(message.message_type)
        # The metadata leaves no empty container behind.
        self.assertNotIn(CUSTOM, message)

    def test_add_user_accepts_a_message_type(self):
        conversation = Conversation()
        message = conversation.add_user(
            'hi', message_type=ConversationMessageType.IMPERSONATION)
        self.assertEqual(message.message_type,
                         ConversationMessageType.IMPERSONATION)
        self.assertEqual(conversation.to_openai(),
                         [{'role': 'user', 'content': 'hi'}])

    def test_unknown_persisted_tag_is_not_a_known_type(self):
        # A hand-edited or older persisted history can carry a tag value
        # outside the current enum. It reads as None - not coerced to a
        # known type - so nothing fails on load and no comparison misfires.
        message = ConversationMessage({
            'role': 'user', 'content': 'hi',
            CUSTOM: {'message_type': 'defunct-kind'}})
        self.assertIsNone(message.message_type)
        self.assertFalse(is_thinking_message(message))


class TestConversationAddImage(unittest.TestCase):

    def test_only_the_most_recent_max_images_are_kept(self):
        # The cap is a per-conversation constructor parameter; both an
        # explicit small cap and the default are honoured by pruning.
        for max_images in (2, MAX_IMAGES):
            with self.subTest(max_images=max_images):
                conversation = Conversation(max_images=max_images)
                self.assertEqual(conversation.max_images, max_images)
                for n in range(max_images + 4):
                    conversation.add_image(bytes([n]), 'image/png',
                                           f'img {n}')
                image_urls = [part['image_url']['url']
                              for message in conversation
                              for part in message['content']
                              if part['type'] == 'image_url']
                self.assertEqual(len(image_urls), max_images)
                # The four oldest images merged their placeholder into
                # the caption.
                self.assertEqual(
                    [message['content'] for message in conversation[:4]],
                    [[{'type': 'text', 'text': f'img {n}\n{IMAGE_PLACEHOLDER}'}]
                     for n in range(4)])

    def test_max_images_below_one_is_rejected(self):
        # A cap of 0 cannot be honoured: the prune runs before the append,
        # so the image being added would always survive and the cap would
        # silently end at 1. Nonsensical caps are refused outright.
        for bad in (0, -1):
            with self.assertRaisesRegex(ValueError,
                                        'max_images must be at least 1'):
                Conversation(max_images=bad)

    def test_add_image_appends_a_caption_and_base64_image(self):
        b64 = base64.b64encode(b'abc').decode()
        for caption, mime, expected in [
            ('look at this', 'image/png',
             {'role': 'user',
              'content': [
                  {'type': 'text', 'text': 'look at this'},
                  {'type': 'image_url',
                   'image_url': {'url': f'data:image/png;base64,{b64}'}},
              ]}),
            (None, 'image/jpeg',
             {'role': 'user',
              'content': [
                  {'type': 'image_url',
                   'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
              ]}),
        ]:
            with self.subTest(caption=caption):
                conversation = Conversation()
                message = conversation.add_image(b'abc', mime, caption)
                self.assertEqual(message, expected)

    def test_pruning_leaves_messages_without_images_untouched(self):
        conversation = Conversation()
        conversation.add_user('plain text turn')
        conversation.add_assistant('reply')
        for n in range(conversation.max_images + 4):
            conversation.add_image(bytes([n]), 'image/png', f'img {n}')
        self.assertEqual(conversation[0], {'role': 'user',
                                           'content': 'plain text turn'})
        self.assertEqual(conversation[1], {'role': 'assistant',
                                           'content': 'reply'})
        self.assertEqual(len(conversation), conversation.max_images + 6)

    def test_text_sharing_a_message_with_a_pruned_image_is_kept(self):
        conversation = Conversation()
        conversation.add_user([
            {'type': 'text', 'text': 'what is this?'},
            {'type': 'image_url',
             'image_url': {'url': 'data:image/png;base64,AAA'}},
        ])
        for n in range(conversation.max_images + 2):
            conversation.add_image(bytes([n]), 'image/png', f'img {n}')
        self.assertEqual(conversation[0]['content'], [
            {'type': 'text', 'text': f'what is this?\n{IMAGE_PLACEHOLDER}'},
        ])

    def test_pruned_image_without_caption_becomes_the_placeholder(self):
        conversation = Conversation()
        conversation.add_image(b'x', 'image/png')
        for n in range(conversation.max_images + 2):
            conversation.add_image(bytes([n]), 'image/png')
        self.assertEqual(conversation[0]['content'],
                         [{'type': 'text', 'text': IMAGE_PLACEHOLDER}])


class TestOpenAiContract(unittest.TestCase):
    """validate_openai_contract: the invariants to_openai() output relies
    on - one system head, and tool calls paired with their results - are
    checkable on demand, never automatically on the request path."""

    def test_a_well_formed_conversation_has_no_problems(self):
        conversation = Conversation()
        conversation.set_system_head('prompt')
        conversation.add_user('go')
        assistant = conversation.add_assistant('')
        assistant['tool_calls'] = [
            {'id': 'c1', 'type': 'function',
             'function': {'name': 'a', 'arguments': '{}'}},
            {'id': 'c2', 'type': 'function',
             'function': {'name': 'b', 'arguments': '{}'}},
        ]
        conversation.add_tool('one', 'c1')
        conversation.add_tool('two', 'c2')
        conversation.add_assistant('done')
        self.assertEqual(conversation.validate_openai_contract(), [])

    def test_more_than_one_system_message_is_reported(self):
        conversation = Conversation()
        conversation.set_system_head('prompt')
        conversation.add_user('hi')
        conversation.add('system', 'a second prompt')
        problems = conversation.validate_openai_contract()
        self.assertEqual(len(problems), 1)
        self.assertIn('more than one system message', problems[0])

    def test_a_system_message_off_the_head_is_reported(self):
        conversation = Conversation([{'role': 'user', 'content': 'hi'},
                                     {'role': 'system', 'content': 'p'}])
        problems = conversation.validate_openai_contract()
        self.assertEqual(len(problems), 1)
        self.assertIn('not the head', problems[0])

    def test_an_unanswered_tool_call_is_reported(self):
        conversation = Conversation()
        conversation.add_user('go')
        assistant = conversation.add_assistant('')
        assistant['tool_calls'] = [
            {'id': 'c1', 'type': 'function',
             'function': {'name': 'a', 'arguments': '{}'}},
        ]
        problems = conversation.validate_openai_contract()
        self.assertEqual(problems, ['unanswered tool calls: [\'c1\']'])

    def test_a_tool_result_answering_nothing_is_reported(self):
        conversation = Conversation()
        conversation.add_user('go')
        conversation.add_tool('orphan', 'c9')
        problems = conversation.validate_openai_contract()
        self.assertEqual(len(problems), 1)
        self.assertIn('answers no open call', problems[0])
        self.assertIn('c9', problems[0])

    def test_thinking_blocks_are_not_contract_problems(self):
        # Thinking is stripped by to_openai(), never sent, so it cannot
        # break the endpoint contract even though it sits in the history.
        conversation = Conversation()
        conversation.set_system_head('prompt')
        conversation.add_user('go')
        conversation.add_thinking()
        conversation.add_assistant('done')
        self.assertEqual(conversation.validate_openai_contract(), [])


class TestConversationListCompat(unittest.TestCase):

    def test_two_conversations_with_same_messages_are_equal_but_distinct_ids(self):  # noqa: E501
        messages = [_user('hello'), {'role': 'assistant', 'content': 'hi'}]
        conversation = Conversation(messages)
        self.assertEqual(conversation, messages)
        self.assertNotEqual(conversation, [_user('hello')])
        a = Conversation(list(messages))
        b = Conversation(list(messages))
        self.assertEqual(a, b)
        self.assertNotEqual(a.id, b.id)


class TestThinkingMessages(unittest.TestCase):
    """Kept thinking blocks: first-class history, display-only for the API."""

    def test_add_thinking_tags_an_assistant_message(self):
        conversation = Conversation()
        message = conversation.add_thinking(telegram_message_ids=[7])
        self.assertEqual(message.get('role'), 'assistant')
        self.assertEqual(message.get('content'), '')
        self.assertTrue(is_thinking_message(message))
        self.assertEqual(message.telegram_message_ids, [7])

    def test_to_openai_skips_thinking(self):
        conversation = Conversation()
        conversation.add_user('hi')
        conversation.add_thinking()
        conversation.add_assistant('final')
        self.assertEqual(conversation.to_openai(),
                         [{'role': 'user', 'content': 'hi'},
                          {'role': 'assistant', 'content': 'final'}])

    def test_unknown_tag_is_treated_as_not_thinking(self):
        # An unknown message_type value is never coerced to a known
        # type: a hand-edited or older persisted history keeps loading
        # on a stale tag, and such a message cannot be display-only,
        # so it reaches the endpoint like any message.
        message = {'role': 'assistant', 'content': 'hi',
                   CUSTOM: {'message_type': 'defunct-kind'}}
        self.assertFalse(is_thinking_message(message))
        self.assertEqual(Conversation([message]).to_openai(),
                         [{'role': 'assistant', 'content': 'hi'}])

    def test_token_estimate_skips_thinking(self):
        conversation = Conversation()
        conversation.add_user('hi')
        conversation.add_thinking()
        self.assertEqual(conversation.token_estimate,
                         estimate_message_tokens(
                             conversation.messages[0]))

    def test_undo_removes_thinking_with_the_exchange(self):
        conversation = Conversation()
        conversation.add_user('q', telegram_message_ids=[1])
        conversation.add_thinking(telegram_message_ids=[2])
        conversation.add_assistant('a', telegram_message_ids=[3])
        removed = conversation.undo_last_exchange()
        self.assertEqual(conversation.messages, [])
        self.assertEqual(len(removed), 3)

    def test_prepare_regeneration_drops_thinking_and_answer(self):
        for label, with_answer in (('thinking and answer', True),
                                   ('orphan thinking', False)):
            with self.subTest(exchange=label):
                conversation = Conversation()
                conversation.add_user('q')
                conversation.add_thinking(telegram_message_ids=[2])
                if with_answer:
                    conversation.add_assistant('a', telegram_message_ids=[3])
                removed, action = conversation.prepare_regeneration()
                self.assertEqual(action, RegenerationAction.ANSWER)
                self.assertEqual(
                    [m.get('role') for m in conversation.messages], ['user'])
                if with_answer:
                    ids = sorted(i for m in removed
                                 for i in m.telegram_message_ids)
                    self.assertEqual(ids, [2, 3])
                else:
                    self.assertEqual(len(removed), 1)

    def test_clone_for_impersonation_has_a_fresh_id(self):
        conversation = Conversation()
        conversation.add_user('hi')
        clone = conversation.clone_for_impersonation('prompt')
        self.assertNotEqual(clone.id, conversation.id)
        self.assertEqual(clone.messages[-1]['content'], 'prompt')
        self.assertEqual(len(conversation.messages), 1)


if __name__ == '__main__':
    unittest.main()
