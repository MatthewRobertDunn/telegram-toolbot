import base64
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation import Conversation, IMAGE_PLACEHOLDER


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


class TestConversationListCompat(unittest.TestCase):

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
