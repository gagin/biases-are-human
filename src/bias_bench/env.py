from pathlib import Path

from dotenv import load_dotenv
import os

# Load .env from project root (two levels up from this file: src/bdb/env.py)
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key
