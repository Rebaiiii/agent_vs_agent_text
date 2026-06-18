"""Minimal OpenAI-compatible chat client for LLMAgent.

Uses only the Python standard library. The client returns the model's raw
message content, which LLMAgent expects to be JSON text.
"""

import json
import urllib.error
import urllib.request

from config import get_llm_config


def make_openai_client(llm_config=None):
    config = llm_config or get_llm_config()
    api_key = config["api_key"]
    model = config["model"]
    base_url = config["base_url"].rstrip("/")

    def client_func(prompt):
        if not api_key or api_key == "your_api_key_here":
            raise RuntimeError("OPENAI_API_KEY is not set in .env")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an agent controller. Return only valid JSON "
                        "with an action and reason."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API request failed: {error.code} {error_body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM API request failed: {error}") from error

        return data["choices"][0]["message"]["content"]

    return client_func
