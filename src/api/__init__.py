"""
src.api - Kie.ai generation clients (kie_client, hero_generator).

Importing this package auto-loads the project-root `.env` so KIE_API_KEY is
available without any manual `export`. A key already set in the shell wins
(so the old `export KIE_API_KEY=...` workflow still works untouched).
No external dependency - a tiny hand-rolled .env parser.
"""
import os as _os
from pathlib import Path as _Path


def _load_dotenv():
    # Respect a key already exported in the shell - don't override it.
    if _os.getenv("KIE_API_KEY"):
        return
    env_file = _Path(__file__).resolve().parents[2] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        _os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
