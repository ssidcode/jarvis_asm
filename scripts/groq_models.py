"""List the Groq models your API key can actually use.

    python -m scripts.groq_models
    python -m scripts.groq_models --test      # also send a 1-token live request

Groq retires models on a few months' notice, so a model ID that worked last
quarter can start returning HTTP 404 with nothing having changed on your
machine. This asks the API directly, which is the only authoritative answer —
docs and third-party guides go stale.
"""
from __future__ import annotations

import argparse
import os
import sys

# Models whose IDs look like speech rather than chat.
SPEECH_HINTS = ("whisper", "tts", "orpheus", "distil")
GUARD_HINTS = ("guard", "safeguard", "prompt-guard")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List available Groq models")
    parser.add_argument("--api-key-env", default="GROQ_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--test", action="store_true",
                        help="send a tiny real request to each chat model")
    args = parser.parse_args(argv)

    try:
        import httpx
    except ImportError:
        print("httpx is required:  pip install httpx", file=sys.stderr)
        return 2

    from jarvis.llm import GROQ_BASE_URL

    base = (args.base_url or GROQ_BASE_URL).rstrip("/")
    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        print(f"{args.api_key_env} is not set.\n"
              f"  export {args.api_key_env}=gsk_...   (no space after '=')",
              file=sys.stderr)
        return 2

    headers = {"Authorization": f"Bearer {key}"}
    try:
        with httpx.Client(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
            response = client.get(f"{base}/models", headers=headers)
            if response.status_code >= 400:
                print(f"HTTP {response.status_code}: {response.text[:300]}", file=sys.stderr)
                if response.status_code in (401, 403):
                    print("\n-> the key was rejected; check it is current and "
                          "has no stray whitespace", file=sys.stderr)
                return 1
            payload = response.json()
    except Exception as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    models = sorted(
        (m.get("id", "") for m in payload.get("data", []) if m.get("id")),
    )
    if not models:
        print("the API returned no models for this key")
        return 1

    chat, speech, guard = [], [], []
    for name in models:
        low = name.lower()
        if any(h in low for h in SPEECH_HINTS):
            speech.append(name)
        elif any(h in low for h in GUARD_HINTS):
            guard.append(name)
        else:
            chat.append(name)

    print(f"\n{len(models)} model(s) available to this key\n")
    print("CHAT  (use one for llm.model in config.yaml)")
    for name in chat:
        print(f"    {name}")
    if speech:
        print("\nSPEECH  (use one for stt.groq_model)")
        for name in speech:
            print(f"    {name}")
    if guard:
        print("\nMODERATION")
        for name in guard:
            print(f"    {name}")

    if args.test and chat:
        print("\nlive check (1 token each):")
        with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            for name in chat:
                body = {
                    "model": name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_completion_tokens": 1,
                }
                try:
                    reply = client.post(
                        f"{base}/chat/completions", headers=headers, json=body
                    )
                    mark = "ok  " if reply.status_code < 400 else f"{reply.status_code} "
                except Exception as exc:
                    mark = f"err ({type(exc).__name__})"
                print(f"    [{mark}] {name}")

    print("\nSet your choice in config.yaml:")
    print("  llm:\n    model: <one of the CHAT ids above>")
    print("  stt:\n    groq_model: <one of the SPEECH ids above>\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())