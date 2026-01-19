import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


HISTORY_FILE = Path(__file__).resolve().parent.parent / "history.json"
_lock = threading.Lock()
MAX_HISTORY_ITEMS = 10


def _load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        # In case of a corrupted file, start fresh instead of crashing.
        return []
    return []


def _save_history(items: List[Dict[str, Any]]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def add_history_entry(entry_type: str, payload: Dict[str, Any]) -> None:
    """
    Add a single history entry.
    entry_type: 'text', 'image', or 'parse_demo'
    payload: arbitrary dict (we will store a compact subset to avoid huge files)
    """
    with _lock:
        items = _load_history()
        items.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "type": entry_type,
                "payload": payload,
            }
        )
        # Keep only the last MAX_HISTORY_ITEMS entries.
        items = items[-MAX_HISTORY_ITEMS:]
        _save_history(items)


def get_history() -> List[Dict[str, Any]]:
    """
    Return the last history entries (up to MAX_HISTORY_ITEMS).
    """
    with _lock:
        return _load_history()

