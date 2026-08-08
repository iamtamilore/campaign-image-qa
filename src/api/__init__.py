import os as _os
from pathlib import Path as _Path


def _load_dotenv():
    # Respect a key already exported in the shell, don't override it.
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
