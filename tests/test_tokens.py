import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import compaction
from tokens import (IMAGE_TOKEN_ESTIMATE, TOKENS_PER_CHAR, estimate_message_tokens,
                    token_count)


def _text(tokens: int) -> str:
    """A string that token_count() estimates at exactly `tokens` tokens."""
    return 'x' * int(tokens / TOKENS_PER_CHAR)


class TestTokenCount(unittest.TestCase):

    def test_quarter_token_per_character(self):
        self.assertEqual(token_count(''), 0)
        self.assertEqual(token_count('x' * 400), 100)

    def test_estimates_text_and_images(self):
        conversation = [
            {'role': 'system', 'content': 'x' * 400},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'x' * 400},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
            ]},
        ]
        self.assertEqual(
            compaction.estimate_conversation_tokens(conversation),
            100 + 100 + IMAGE_TOKEN_ESTIMATE)

    def test_estimates_tool_calls(self):
        message = {
            'role': 'assistant', 'content': '',
            'tool_calls': [{
                'id': 'call_1', 'type': 'function',
                'function': {'name': 'toolname', 'arguments': 'x' * 400},
            }],
        }
        self.assertEqual(estimate_message_tokens(message), 102)


if __name__ == '__main__':
    unittest.main()
