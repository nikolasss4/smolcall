"""Call the fine-tune through mlx_lm.server's OpenAI-compatible endpoint.

    uv run mlx_lm.server --model models/smolcall-4bit --port 8080
    uv run python examples/server_client.py "Convert 50 USD to EUR"

The server is plain chat completions, so the client builds smolcall's system prompt
(tools as JSON Schema) and parses the JSON array that comes back. Standard library only.
"""

import json
import sys
import urllib.request
from pathlib import Path

from smolcall.eval.scorer import parse_output
from smolcall.prompt import system_prompt

URL = "http://127.0.0.1:8080/v1/chat/completions"


def main(query: str) -> None:
    tools = json.loads(Path("examples/tools.json").read_text())
    body = {
        "messages": [
            {"role": "system", "content": system_prompt(tools)},
            {"role": "user", "content": query},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        text = json.load(resp)["choices"][0]["message"]["content"]
    calls = parse_output(text)
    print(json.dumps([c.__dict__ for c in calls] if calls is not None else None, indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "What's the weather in Oslo, in fahrenheit?")
