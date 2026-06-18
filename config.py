"""Small .env loader for local LLM API settings.

This keeps the project dependency-free. It supports simple KEY=value lines.
"""

import os
from pathlib import Path


def load_env(path=".env"):
    env_path = Path(path)
    if not env_path.is_absolute():
        env_path = Path(__file__).resolve().parent / env_path

    if not env_path.exists():
        return {}

    loaded = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        loaded[key] = os.environ[key]

    return loaded


def get_llm_config():
    load_env()
    return {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "evaluate_llm": os.environ.get("EVALUATE_LLM", "false").lower() == "true",
        "llm_eval_games": int(os.environ.get("LLM_EVAL_GAMES", "5")),
    }
