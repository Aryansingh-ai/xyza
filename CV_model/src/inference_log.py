"""Best-effort JSONL logging for intentional inference events."""

import json
from pathlib import Path


def append_inference_log(result: dict, log_path: str) -> bool:
    """Append one result to JSONL; never let logging stop inference."""
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        return True
    except (OSError, TypeError, ValueError) as error:
        print(f"Warning: could not append inference log '{log_path}': {error}")
        return False
