# Verification commands (pake tasks)

All commands are pake tasks defined in `pakefile.py` ([pake](https://github.com/Teriks/pake).

```powershell
pake typecheck         # pyright over src/ tests/ integration_tests/
pake unittest          # unit tests (pure logic, no network)
pake integration_test  # integration tests (requires OPENAI_API_KEY in integration_tests/.env)
pake publish              # scp src/ to matty@derg.top:/home/matty/tbot, restart tbot user service
pake publish_env          # copy the local .env to the deployment server, if one exists
pake publish_requirements # copy requirements.txt to the deployment server and pip install it into the tbot venv
pake update_sandbox_image  # docker pull the latest sandbox image on the deployment server
```

# Build tasks (pake)

Build tasks are defined in `pakefile.py` using [pake](https://github.com/Teriks/pake).

```powershell
pake build_sandbox_image            # Build the sandbox image for the local platform
pake push_sandbox_image             # Build + push multi-arch image (amd64 + arm64) via buildx
pake -D PLATFORMS=linux/amd64 push_sandbox_image   # Override target platforms
pake -t                             # List tasks
```

# Conventions

- Prefer relative imports within a package (e.g. `from .draft_streamer import DraftStreamer`, not `from handlers.draft_streamer import ...`).

