"""integration_tests/.env must be visible to config, and never win the DB.

``bootstrap.py`` loads ``integration_tests/.env`` BEFORE the shared
harness initialises the app config, because ``config.py`` snapshots the
environment once at import time. These tests pin that order in a fresh
interpreter (the outer process has long since imported config):

* a key that exists only in ``integration_tests/.env`` reaches the
  ``config`` constants (the providers branch's ``requires_nvidia_key``
  decorators rely on this),
* a ``STORE_DB_PATH`` value in that .env file does NOT win — the
  isolated per-run test database does.
"""

import os
import subprocess
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_HERE, '.env')

_PROBE = (
    "import os\n"
    "import sys\n"
    "\n"
    "# Simulate 'key exists only in integration_tests/.env': the\n"
    "# surrounding environment must not already provide it.\n"
    "os.environ.pop('NVIDIA_API_KEY', None)\n"
    "os.environ.pop('OPENAI_API_KEY', None)\n"
    "os.environ.pop('TOOLBOT_BOOTSTRAP_SENTINEL', None)\n"
    "os.environ.pop('STORE_DB_PATH', None)\n"
    "\n"
    f"sys.path.insert(0, {_HERE!r})\n"
    "import bootstrap  # noqa: E402  (.env first, then the DB harness)\n"
    "import config  # noqa: E402\n"
    "from tests import _bootstrap as harness\n"
    "\n"
    "assert os.environ.get('TOOLBOT_BOOTSTRAP_SENTINEL') == '1', (\n"
    "    'sentinel from integration_tests/.env missing from os.environ')\n"
    "assert config.NVIDIA_API_KEY == 'sentinel-nv-key', (\n"
    "    f'config constants missed the .env value: '\n"
    "    f'{config.NVIDIA_API_KEY!r}')\n"
    "assert config.OPENAI_KEY == 'sentinel-openai-key', config.OPENAI_KEY\n"
    "# The .env file poisons STORE_DB_PATH; the isolated DB must win.\n"
    "assert config.sql_db_path == harness.db_path(), (\n"
    "    f'.env STORE_DB_PATH leaked into config: '\n"
    "    f'{config.sql_db_path!r}')\n"
    "assert 'test_store.db' not in config.sql_db_path\n"
    "print('probe-ok')\n"
)


class TestBootstrapEnvOrder(unittest.TestCase):
    """The ordering guarantees, proven in a fresh interpreter."""

    def test_env_keys_reach_config_and_isolated_db_wins(self):
        backup = None
        if os.path.exists(_ENV_FILE):
            with open(_ENV_FILE, encoding='utf-8') as f:
                backup = f.read()
        # Registered first so it runs LAST (cleanups are LIFO): it checks
        # the state the restore below left behind.
        self.addCleanup(self._assert_restored, backup)
        self.addCleanup(self._restore, backup)
        with open(_ENV_FILE, 'w', encoding='utf-8') as f:
            f.write('TOOLBOT_BOOTSTRAP_SENTINEL=1\n'
                    'NVIDIA_API_KEY=sentinel-nv-key\n'
                    'OPENAI_API_KEY=sentinel-openai-key\n'
                    # Poison: must be overridden by the harness.
                    'STORE_DB_PATH=integration_tests/test_store.db\n')

        result = subprocess.run(
            [sys.executable, '-c', _PROBE],
            capture_output=True, text=True, timeout=120, cwd=_HERE)

        self.assertEqual(
            result.returncode, 0,
            f'probe failed:\n{result.stdout}\n{result.stderr}')
        self.assertIn('probe-ok', result.stdout)

    @staticmethod
    def _restore(backup: str | None) -> None:
        if backup is None:
            os.unlink(_ENV_FILE)
        else:
            with open(_ENV_FILE, 'w', encoding='utf-8') as f:
                f.write(backup)

    @staticmethod
    def _assert_restored(backup: str | None) -> None:
        # The fixture must not leak into the developer's checkout.
        if backup is None:
            assert not os.path.exists(_ENV_FILE), _ENV_FILE
        else:
            with open(_ENV_FILE, encoding='utf-8') as f:
                assert f.read() == backup


if __name__ == '__main__':
    unittest.main()
