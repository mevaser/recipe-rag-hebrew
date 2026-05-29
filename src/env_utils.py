from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def find_project_root(start_path: str | Path | None = None) -> Path:
    current = Path(start_path or Path(__file__).resolve()).resolve()
    search_roots = [current, *current.parents]
    for candidate in search_roots:
        if (
            (candidate / ".env").exists()
            or (candidate / "README.md").exists()
            or (candidate / "src").is_dir()
        ):
            return candidate
    return Path(__file__).resolve().parents[1]


def dotenv_path(start_path: str | Path | None = None) -> Path:
    return find_project_root(start_path) / ".env"


def load_project_env(start_path: str | Path | None = None) -> bool:
    env_path = dotenv_path(start_path)
    if env_path.exists():
        load_dotenv(env_path, override=False)
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def print_openai_api_key_status(start_path: str | Path | None = None) -> bool:
    loaded = load_project_env(start_path)
    print(f"OPENAI_API_KEY loaded: {'yes' if loaded else 'no'}")
    return loaded
