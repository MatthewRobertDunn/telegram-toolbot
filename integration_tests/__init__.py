import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
