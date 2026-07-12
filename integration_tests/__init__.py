import os
import sys

# Needed so `from bootstrap import ...` / `from mocks import ...` work in
# dotted-path imports like `python -m unittest integration_tests.test_commands`.
# discover -s handles this itself; this covers the other mode.
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
