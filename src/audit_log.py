import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STORE_PATH = DATA_DIR / "logs.json"
MAX_LOGS = 5000


def _ensure_store() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        store = {"last_id": 0, "logs": []}
        STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")
        return store
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"last_id": 0, "logs": []}


def _save_store(store: Dict[str, Any]) -> None:
    # Trim to MAX_LOGS (keep newest)
    logs = store.get("logs", [])
    if len(logs) > MAX_LOGS:
        store["logs"] = logs[-MAX_LOGS:]
    STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _format_log_id(n: int) -> str:
    return f"LG-{n:06d}"


def append_log(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Append an audit log entry and return it with assigned id/timestamp."""
    store = _ensure_store()
    store["last_id"] = int(store.get("last_id", 0)) + 1
    entry = dict(entry or {})
    entry["id"] = _format_log_id(store["last_id"])
    entry.setdefault("time", datetime.utcnow().isoformat() + "Z")
    store.setdefault("logs", []).append(entry)
    _save_store(store)
    return entry


def list_logs(limit: int = 200) -> List[Dict[str, Any]]:
    store = _ensure_store()
    logs = store.get("logs", [])
    return logs[-limit:][::-1]


