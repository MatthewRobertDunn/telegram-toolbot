import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation import Conversation
from transcript import message_to_text


class TestMessageToText(unittest.TestCase):

    def test_system_messages_are_left_out(self):
        self.assertEqual(
            message_to_text({'role': 'system', 'content': 'prompt'}),
            '')

    def test_images_become_placeholder_text(self):
        text = message_to_text({
            'role': 'user', 'content': [
                {'type': 'text', 'text': 'look at this'},
                {'type': 'image_url',
                 'image_url': {'url': 'data:image/png;base64,AAAA'}},
            ],
        })
        self.assertEqual(text, '[User]: look at this\n[image omitted]')
        self.assertNotIn('base64', text)

    def test_assistant_tool_calls(self):
        text = message_to_text({
            'role': 'assistant', 'content': 'checking',
            'tool_calls': [{
                'id': 'call_1', 'type': 'function',
                'function': {'name': 'read', 'arguments': '{"path": "a.txt"}'},
            }],
        })
        self.assertEqual(
            text,
            '[Assistant]: checking\n'
            '[Assistant tool call]: read({"path": "a.txt"})')

    def test_tool_output_is_truncated(self):
        text = message_to_text(
            {'role': 'tool', 'content': 'y' * 5_000})
        self.assertTrue(text.startswith('[Tool result]: '))
        self.assertTrue(text.endswith('\n[truncated]'))
        self.assertLess(len(text), 5_000)

    def test_thinking_blocks_are_left_out(self):
        conversation = Conversation()
        thinking = conversation.add_thinking()
        self.assertEqual(message_to_text(thinking), '')


if __name__ == '__main__':
    unittest.main()
