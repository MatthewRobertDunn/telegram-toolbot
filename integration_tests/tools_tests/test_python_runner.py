import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bootstrap  # noqa: F401

from tools.python_runner import run_python_code


class TestPythonRunner(unittest.IsolatedAsyncioTestCase):

    async def test_runs_code_and_returns_stdout(self):
        result = await run_python_code("print(2 + 2)")
        self.assertIn("Return code: 0", result)
        self.assertIn("stdout:", result)
        self.assertIn("4", result)

    async def test_returns_stderr_output(self):
        result = await run_python_code("import sys; sys.stderr.write('oops')")
        self.assertIn("stderr:", result)
        self.assertIn("oops", result)

    async def test_nonzero_return_code(self):
        result = await run_python_code("raise ValueError('boom')")
        self.assertIn("Return code: 1", result)
        self.assertIn("boom", result)


if __name__ == "__main__":
    unittest.main()
