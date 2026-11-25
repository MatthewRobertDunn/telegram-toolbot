"""Environment parsing: a bad value fails naming the variable, at import.

config is reloaded with dotenv suppressed so the real .env cannot leak
values into these assertions.
"""

import importlib
import os
import re
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

    # (env variable, bad raw value, parse helper exercised at import)
    BAD_ENV_VALUES = [
        ('SANDBOX_PIDS_LIMIT', 'a lot', config._int_env),
        ('SANDBOX_EXEC_TIMEOUT', 'soon', config._float_env),
        ('API_RETRY_TIME', 'forever', config._int_env),
        ('API_RETRY_MAX_DELAY', 'a while', config._float_env),
        ('ALLOWED_IDS', '1, two, three', config._int_list_env),
    ]

    def test_bad_values_name_the_variable_and_value(self):
        for variable, raw, parse_helper in self.BAD_ENV_VALUES:
            with self.subTest(variable=variable,
                              helper=parse_helper.__name__):
                with self.assertRaises(ValueError) as caught:
                    load_with_env(**{variable: raw})
                message = str(caught.exception)
                self.assertIn(variable, message)
                self.assertIn(raw, message)

    def test_numeric_settings_are_env_tunable(self):
        load_with_env(TEMPERATURE='0.5', TOP_P='0.4', API_RETRY_TIME='42',
                      API_RETRY_MAX_DELAY='30', SANDBOX_PIDS_LIMIT='64')
        self.assertEqual(config.TEMPERATURE, 0.5)
        self.assertEqual(config.TOP_P, 0.4)
        self.assertEqual(config.API_RETRY_TIME, 42)
        self.assertEqual(config.API_RETRY_MAX_DELAY, 30)
        self.assertEqual(config.SANDBOX_PIDS_LIMIT, 64)
        # The lowercase alias names stay in step.
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
        # There is deliberately no STORE_DB_PATH module attribute: a
        # one-time copy would silently ignore patches and drift from
        # sql_db_path, which is what the stores read at call time.
        self.assertFalse(hasattr(config, 'STORE_DB_PATH'))

    # (scenario, env to reload under, expected NVIDIA_API_KEY, whether
    # the fallback to OPENAI_API_KEY is announced in a warning)
    NVIDIA_KEY_SCENARIOS = [
        # The README's single-key deployment keeps working.
        ('fallback', {'OPENAI_API_KEY': 'sk-openai'}, 'sk-openai', True),
        ('explicit key wins', {'NVIDIA_API_KEY': 'nvsk-test',
                               'OPENAI_API_KEY': 'sk-openai'},
         'nvsk-test', False),
        # Nothing to fall back to: no key and no misleading warning.
        ('no keys', {}, '', False),
    ]

    def test_nvidia_key_is_resolved_and_the_fallback_announced(self):
        for label, env, expected_key, announced in self.NVIDIA_KEY_SCENARIOS:
            with self.subTest(scenario=label):
                if announced:
                    with self.assertLogs('config', level='WARNING') as logs:
                        load_with_env(**env)
                    message = "\n".join(logs.output)
                    self.assertIn('NVIDIA_API_KEY', message)
                    self.assertIn('OPENAI_API_KEY', message)
                else:
                    with self.assertNoLogs('config', level='WARNING'):
                        load_with_env(**env)
                self.assertEqual(config.NVIDIA_API_KEY, expected_key)


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

    # (scenario, where among parent/root/src a .env is written, expected
    # result: the project root's .env or None)
    ENV_FILE_SCENARIOS = [
        ('root env found', {'root'}, 'root'),
        ('root env wins over a stale src env', {'root', 'src'}, 'root'),
        # A .env above the project root (e.g. a home directory) must not
        # be loaded.
        ('ancestor env ignored', {'parent'}, None),
        ('no env anywhere', set(), None),
    ]

    def test_only_the_project_root_env_is_considered(self):
        for label, env_dirs, expected in self.ENV_FILE_SCENARIOS:
            with self.subTest(scenario=label):
                with tempfile.TemporaryDirectory() as parent:
                    root = os.path.join(parent, 'project')
                    src = os.path.join(root, 'src')
                    os.makedirs(src)
                    places = {'parent': parent, 'root': root, 'src': src}
                    for where in env_dirs:
                        self._write_env(places[where])
                    with self._config_in(src):
                        found = config._find_env_file()
                if expected is None:
                    self.assertIsNone(found)
                else:
                    self.assertEqual(found,
                                     os.path.join(root, '.env'))


class ConfigReferenceTest(unittest.TestCase):
    """Every ``config.<name>`` in src/ must exist on the config module.

    Nothing imports config with a type checker in the loop, so a
    reference to a removed or never-added name only explodes when that
    line runs - often in a rarely exercised error path. This scan pins
    the whole surface instead.
    """

    # Prose like "see config.py" is a file name, not an attribute.
    FILE_NAME_WORDS = {'py'}

    def test_every_referenced_name_exists(self):
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'src')
        pattern = re.compile(r'\bconfig\.([A-Za-z_][A-Za-z0-9_]*)')
        missing = []
        for directory, _, filenames in os.walk(src):
            for filename in filenames:
                if not filename.endswith('.py'):
                    continue
                path = os.path.join(directory, filename)
                with open(path, encoding='utf-8') as handle:
                    for lineno, line in enumerate(handle, 1):
                        for name in pattern.findall(line):
                            if name in self.FILE_NAME_WORDS:
                                continue
                            if not hasattr(config, name):
                                missing.append(
                                    f"{os.path.relpath(path, src)}:"
                                    f"{lineno}: config.{name}")
        self.assertEqual(missing, [],
                         'unknown config attributes referenced:\n  '
                         + '\n  '.join(missing))


if __name__ == '__main__':
    unittest.main()
