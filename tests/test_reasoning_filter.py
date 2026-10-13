import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from handlers.run_with_tools import _split_thinking

# Build the tag literals here so nothing in this file can strip them.
TAG_OPEN = '<' + 'think' + '>'
TAG_CLOSE = '</' + 'think' + '>'


class TestSplitThinking(unittest.TestCase):

    def test_no_tags(self):
        visible, reasoning = _split_thinking("Hello world.")
        self.assertEqual(visible, "Hello world.")
        self.assertEqual(reasoning, '')

    def test_full_span(self):
        text = TAG_OPEN + "Let me check." + TAG_CLOSE + "Hi there!"
        visible, reasoning = _split_thinking(text)
        self.assertEqual(visible, "Hi there!")
        self.assertIn("Let me check.", reasoning)

    def test_multiline_reasoning(self):
        text = (TAG_OPEN + "line one"
                + chr(13) + chr(10) + "line two"
                + TAG_CLOSE + "Answer.")
        visible, reasoning = _split_thinking(text)
        self.assertEqual(visible, "Answer.")
        self.assertIn("line one", reasoning)

    def test_lone_close_tag_before_content(self):
        text = TAG_CLOSE + "Ready."
        visible, reasoning = _split_thinking(text)
        self.assertEqual(visible, "Ready.")
        self.assertEqual(reasoning, '')

    def test_no_open_tag_bare_close(self):
        # backends that emit reasoning with no open tag, ending in a close
        text = "plan the reply" + TAG_CLOSE + "Final answer."
        visible, reasoning = _split_thinking(text)
        self.assertEqual(visible, "Final answer.")
        self.assertEqual(reasoning, "plan the reply")

    def test_unclosed_tag_left_alone(self):
        # no close tag anywhere: nothing can be classified as reasoning
        text = TAG_OPEN + "cut short by an error"
        visible, reasoning = _split_thinking(text)
        self.assertEqual(visible, text)
        self.assertEqual(reasoning, '')

    def test_case_sensitive(self):
        text = "<THINK>hidden</THINK>shown"
        visible, reasoning = _split_thinking(text)
        self.assertEqual(visible, text)
        self.assertEqual(reasoning, '')


if __name__ == '__main__':
    unittest.main()
