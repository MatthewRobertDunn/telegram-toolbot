# Verification commands (pake tasks)

All commands are pake tasks defined in `pmakefile.py` ([pake](https://github.com/Teriks/pake).

```powershell
pake typecheck         # pyright over src/ tests/ integration_tests/
pake unittest          # unit tests (pure logic, no network)
pake integration_test  # integration tests (requires OPENAI_API_KEY in integration_tests/.env)
```

# Build tasks (pake)

Build tasks are defined in `pmakefile.py` using [pake](https://github.com/Teriks/pake).

```powershell
pake build_sandbox_image            # Build the sandbox image for the local platform
pake push_sandbox_image             # Build + push multi-arch image (amd64 + arm64) via buildx
pake -D PLATFORMS=linux/amd64 push_sandbox_image   # Override target platforms
pake -t                             # List tasks
```

