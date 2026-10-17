import base64
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation import (Conversation, IMAGE_PLACEHOLDER,
                          estimate_conversation_tokens,
                          estimate_message_tokens, token_count)

# A tool-result message spends tokens on its id-less text only.
def _user(text):
    return {'role': 'user', 'content': text}


class TestConversationIdentity(unittest.TestCase):

    def test_every_conversation_gets_a_unique_id(self):
        self.assertNotEqual(Conversation().id, Conversation().id)

    def test_id_survives_mutations(self):
        conversation = Conversation()
        original_id = conversation.id
        conversation.add_user('hello')
        conversation.add_assistant('hi')
        conversation.set_system_head('prompt')
        self.assertEqual(conversation.id, original_id)

    def test_messages_are_the_first_positional_argument(self):
        conversation = Conversation([_user('hello')])
        self.assertEqual(conversation.messages, [_user('hello')])
        self.assertEqual(conversation[0], _user('hello'))

    def test_default_messages_are_empty(self):
        self.assertEqual(Conversation().messages, [])


class TestConversationHelpers(unittest.TestCase):

    def test_add_returns_the_message(self):
        conversation = Conversation()
        message = conversation.add('user', 'hi', custom=1)
        self.assertEqual(conversation.messages, [message])
        self.assertEqual(message, {'role': 'user', 'content': 'hi',
                                   'custom': 1})

    def test_add_user_and_add_assistant(self):
        conversation = Conversation()
        user_message = conversation.add_user('hello')
        assistant_message = conversation.add_assistant('hi')
        self.assertEqual(conversation.messages,
                         [{'role': 'user', 'content': 'hello'},
                          {'role': 'assistant', 'content': 'hi'}])
        self.assertEqual(user_message['role'], 'user')
        self.assertEqual(assistant_message['role'], 'assistant')

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

    def test_set_system_head_replaces_existing_head(self):
        conversation = Conversation()
        conversation.set_system_head('old')
        conversation.set_system_head('new')
        self.assertEqual(conversation.messages,
                         [{'role': 'system', 'content': 'new'}])

    def test_head_is_system(self):
        conversation = Conversation()
        self.assertFalse(conversation.head_is_system)
        conversation.add_user('hi')
        self.assertFalse(conversation.head_is_system)
        conversation.set_system_head('prompt')
        self.assertTrue(conversation.head_is_system)


class TestConversationAddImage(unittest.TestCase):

    def test_add_image_appends_a_caption_and_base64_image(self):
        conversation = Conversation()
        message = conversation.add_image(b'abc', 'image/png', 'look at this')
        b64 = base64.b64encode(b'abc').decode()
        self.assertEqual(message, {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'look at this'},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/png;base64,{b64}'}},
            ],
        })

    def test_add_image_without_caption_has_just_the_image(self):
        conversation = Conversation()
        message = conversation.add_image(b'abc', 'image/jpeg')
        self.assertEqual(message['content'], [
            {'type': 'image_url',
             'image_url': {'url': f'data:image/jpeg;base64,'
                                  f'{base64.b64encode(b"abc").decode()}'}},
        ])

    def test_only_the_most_recent_max_images_are_kept(self):
        conversation = Conversation()
        for n in range(conversation.max_images + 4):
            conversation.add_image(bytes([n]), 'image/png', f'img {n}')
        image_urls = [part['image_url']['url']
                      for message in conversation
                      for part in message['content']
                      if part['type'] == 'image_url']
        self.assertEqual(len(image_urls), conversation.max_images)
        # The four oldest images merged their placeholder into the caption.
        self.assertEqual(
            [message['content'] for message in conversation[:4]],
            [[{'type': 'text', 'text': f'img {n}\n{IMAGE_PLACEHOLDER}'}]
             for n in range(4)])

    def test_max_images_can_be_shrunk_per_conversation(self):
        conversation = Conversation()
        conversation.max_images = 2
        for n in range(5):
            conversation.add_image(bytes([n]), 'image/png', f'img {n}')
        image_urls = [part['image_url']['url']
                      for message in conversation
                      for part in message['content']
                      if part['type'] == 'image_url']
        self.assertEqual(len(image_urls), 2)
        # The three oldest images merged their placeholder into the caption.
        self.assertEqual(
            [message['content'][0]['text'] for message in conversation[:3]],
            [f'img {n}\n{IMAGE_PLACEHOLDER}' for n in range(3)])

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


class TestConversationTokenEstimate(unittest.TestCase):

    def test_token_estimate_matches_shared_estimators(self):
        conversation = Conversation()
        conversation.add_user('a' * 400)
        conversation.add_assistant('b' * 40)
        conversation.add('user', [{'type': 'text', 'text': 'c' * 100},
                                  {'type': 'image_url',
                                   'image_url': {'url': 'data:image/png'}}])
        self.assertEqual(conversation.token_estimate,
                         estimate_conversation_tokens(conversation.messages))
        self.assertEqual(
            conversation.token_estimate,
            sum(estimate_message_tokens(message)
                for message in conversation.messages))


class TestConversationListCompat(unittest.TestCase):

    def test_len_iter_and_getitem_delegate_to_messages(self):
        conversation = Conversation()
        first = conversation.add_user('hello')
        second = conversation.add_assistant('hi')
        self.assertEqual(len(conversation), 2)
        self.assertEqual(list(conversation), [first, second])
        self.assertIs(conversation[0], first)
        self.assertIs(conversation[-1], second)

    def test_equals_a_message_list_with_the_same_contents(self):
        messages = [_user('hello'), {'role': 'assistant', 'content': 'hi'}]
        conversation = Conversation(messages)
        self.assertEqual(conversation, messages)
        self.assertNotEqual(conversation, [_user('hello')])

    def test_two_conversations_with_same_messages_are_equal_but_distinct_ids(self):  # noqa: E501
        messages = [_user('hello')]
        a = Conversation(list(messages))
        b = Conversation(list(messages))
        self.assertEqual(a, b)
        self.assertNotEqual(a.id, b.id)


if __name__ == '__main__':
    unittest.main()
