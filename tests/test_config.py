"""Environment parsing: a bad value fails naming the variable, at import.

config is reloaded with dotenv suppressed so the real .env cannot leak
values into these assertions.
"""

import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config


def load_with_env(**env) -> None:
    """Re-execute config against exactly the given environment."""
    with mock.patch('dotenv.load_dotenv'), \
            mock.patch.dict(os.environ, env, clear=True):
        importlib.reload(config)


class ConfigEnvTest(unittest.TestCase):

    def tearDown(self):
        # Restore real values for every other test in the process.
        with mock.patch('dotenv.load_dotenv'):
            importlib.reload(config)

    def test_bad_integer_names_the_variable_and_value(self):
        with self.assertRaises(ValueError) as caught:
            load_with_env(SANDBOX_PIDS_LIMIT='a lot')
        message = str(caught.exception)
        self.assertIn('SANDBOX_PIDS_LIMIT', message)
        self.assertIn('a lot', message)

    def test_bad_float_names_the_variable_and_value(self):
        with self.assertRaises(ValueError) as caught:
            load_with_env(SANDBOX_EXEC_TIMEOUT='soon')
        message = str(caught.exception)
        self.assertIn('SANDBOX_EXEC_TIMEOUT', message)
        self.assertIn('soon', message)

    def test_bad_retry_budget_names_the_variable_and_value(self):
        with self.assertRaises(ValueError) as caught:
            load_with_env(API_RETRY_TIME='forever')
        message = str(caught.exception)
        self.assertIn('API_RETRY_TIME', message)
        self.assertIn('forever', message)

    def test_bad_retry_max_delay_names_the_variable_and_value(self):
        with self.assertRaises(ValueError) as caught:
            load_with_env(API_RETRY_MAX_DELAY='a while')
        message = str(caught.exception)
        self.assertIn('API_RETRY_MAX_DELAY', message)
        self.assertIn('a while', message)

    def test_bad_allowlist_names_the_variable_and_value(self):
        with self.assertRaises(ValueError) as caught:
            load_with_env(ALLOWED_IDS='1, two, three')
        message = str(caught.exception)
        self.assertIn('ALLOWED_IDS', message)
        self.assertIn('1, two, three', message)

    def test_numeric_settings_are_env_tunable(self):
        load_with_env(TEMPERATURE='0.5', TOP_P='0.4', API_RETRY_TIME='42',
                      API_RETRY_MAX_DELAY='30', SANDBOX_PIDS_LIMIT='64')
        self.assertEqual(config.TEMPERATURE, 0.5)
        self.assertEqual(config.TOP_P, 0.4)
        self.assertEqual(config.API_RETRY_TIME, 42)
        self.assertEqual(config.API_RETRY_MAX_DELAY, 30)
        self.assertEqual(config.SANDBOX_PIDS_LIMIT, 64)
        # The legacy lowercase names stay in step.
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.4)
        self.assertEqual(config.api_retry_time, 42)
        self.assertEqual(config.api_retry_max_delay, 30)

    def test_defaults_are_unchanged(self):
        load_with_env()
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(config.api_retry_time, 36000)
        self.assertEqual(config.API_RETRY_MAX_DELAY, 300)
        self.assertEqual(config.api_retry_max_delay, 300)
        self.assertEqual(config.SANDBOX_IDLE_STOP_TIMEOUT, 3600)
        self.assertEqual(config.SANDBOX_IDLE_DESTROY_TIMEOUT, 86400)
        self.assertEqual(config.SANDBOX_EXEC_TIMEOUT, 300)
        self.assertEqual(config.SANDBOX_START_TIMEOUT, 300)
        self.assertEqual(config.SANDBOX_REAP_INTERVAL, 60)
        self.assertEqual(config.SANDBOX_PIDS_LIMIT, 256)
        self.assertEqual(config.SANDBOX_MEMORY, '1g')
        self.assertEqual(config.SANDBOX_CPUS, '8.0')
        self.assertEqual(config.ALLOWED_IDS, [])

    def test_store_db_path_env_var_feeds_sql_db_path(self):
        load_with_env(STORE_DB_PATH='/data/agent.db')
        self.assertEqual(config.sql_db_path, '/data/agent.db')

    def test_store_db_path_is_not_a_stale_copy(self):
        # There is deliberately no STORE_DB_PATH module attribute: a
        # one-time copy would silently ignore patches and drift from
        # sql_db_path, which is what the stores read at call time.
        self.assertFalse(hasattr(config, 'STORE_DB_PATH'))

    def test_nvidia_key_falls_back_to_the_openai_key(self):
        # The README's single-key deployment keeps working.
        load_with_env(OPENAI_API_KEY='sk-openai')
        self.assertEqual(config.NVIDIA_API_KEY, 'sk-openai')

    def test_fallback_to_the_openai_key_is_announced(self):
        with self.assertLogs('config', level='WARNING') as logs:
            load_with_env(OPENAI_API_KEY='sk-openai')
        message = "\n".join(logs.output)
        self.assertIn('NVIDIA_API_KEY', message)
        self.assertIn('OPENAI_API_KEY', message)

    def test_explicit_nvidia_key_wins_and_is_not_warned_about(self):
        with self.assertNoLogs('config', level='WARNING'):
            load_with_env(NVIDIA_API_KEY='nvsk-test',
                          OPENAI_API_KEY='sk-openai')
        self.assertEqual(config.NVIDIA_API_KEY, 'nvsk-test')

    def test_no_keys_at_all_is_silent(self):
        # Nothing to fall back to: no key and no misleading warning.
        with self.assertNoLogs('config', level='WARNING'):
            load_with_env()
        self.assertEqual(config.NVIDIA_API_KEY, '')


class FindEnvFileTest(unittest.TestCase):
    """The .env lookup checks the project root and nothing else."""

    @staticmethod
    def _config_in(src: str):
        # Pretend config.py lives in the given src/ directory.
        return mock.patch.object(config, '__file__', os.path.join(src, 'config.py'))

    @staticmethod
    def _write_env(directory: str) -> None:
        with open(os.path.join(directory, '.env'), 'w', encoding='utf-8') as f:
            f.write('SOME_VAR=1\n')

    def test_reads_the_project_root_env(self):
        with tempfile.TemporaryDirectory() as root:
            src = os.path.join(root, 'src')
            os.makedirs(src)
            self._write_env(root)
            with self._config_in(src):
                self.assertEqual(config._find_env_file(),
                                 os.path.join(root, '.env'))

    def test_root_env_wins_over_a_stale_src_env(self):
        with tempfile.TemporaryDirectory() as root:
            src = os.path.join(root, 'src')
            os.makedirs(src)
            self._write_env(root)
            self._write_env(src)
            with self._config_in(src):
                self.assertEqual(config._find_env_file(),
                                 os.path.join(root, '.env'))

    def test_ancestor_env_is_ignored(self):
        # A .env above the project root (e.g. a home directory) must not
        # be loaded.
        with tempfile.TemporaryDirectory() as parent:
            root = os.path.join(parent, 'project')
            src = os.path.join(root, 'src')
            os.makedirs(src)
            self._write_env(parent)
            with self._config_in(src):
                self.assertIsNone(config._find_env_file())

    def test_no_env_anywhere_returns_none(self):
        with tempfile.TemporaryDirectory() as root:
            src = os.path.join(root, 'src')
            os.makedirs(src)
            with self._config_in(src):
                self.assertIsNone(config._find_env_file())


if __name__ == '__main__':
    unittest.main()
