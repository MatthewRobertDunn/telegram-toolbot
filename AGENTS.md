# Verification commands

```powershell
# Type checking
pyright src/ tests/ integration_tests/

# Unit tests (pure logic, no network)
python -m unittest discover -s tests -p "test_*.py" -v

# Integration tests (requires OPENAI_API_KEY in integration_tests/.env)
python -m unittest discover -s integration_tests -p "test_*.py" -v
```
