import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STORE_PATH = DATA_DIR / "logs.json"
MAX_LOGS = 5000

# Optional: Supabase persistence
try:
    from supabase import create_client
    from src.constants import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    _sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY else None
except Exception:
    _sb = None


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
    """Append an audit log entry (Supabase if configured, else JSON)."""
    entry = dict(entry or {})

    # Write to Supabase first, if available
    if _sb:
        try:
            # Map known keys to columns; pass extra context to details jsonb
            # Only include columns that exist in your audit_logs table schema
            columns = {
                "type": entry.get("type"),
                "order_id": entry.get("order_id"),
                # Provide a default to satisfy NOT NULL constraint
                "user_id": entry.get("user_id", "system"),
                "route": entry.get("route"),
                "method": entry.get("method"),
                "status": entry.get("status"),
                "prompt": entry.get("prompt"),
                "tool_name": entry.get("tool_name"),
                "tokens_input": entry.get("tokens_input"),
                "tokens_output": entry.get("tokens_output"),
                "event": entry.get("event"),
                "details": entry.get("details"),
            }
            # Remove None keys to avoid null-overwrites
            row = {k: v for k, v in columns.items() if v is not None}
            res = _sb.table("audit_logs").insert(row).execute()
            # Attach DB id/created_at if present
            if hasattr(res, "data") and res.data:
                entry["db_id"] = res.data[0].get("id")
                entry["time"] = res.data[0].get("created_at")
            return entry
        except Exception as e:
            # Surface error in server logs and fall through to JSON store
            try:
                print(f"[audit_log] Supabase insert failed: {e}")
            except Exception:
                pass

    # JSON fallback (ephemeral)
    store = _ensure_store()
    store["last_id"] = int(store.get("last_id", 0)) + 1
    entry["id"] = _format_log_id(store["last_id"])
    entry.setdefault("time", datetime.utcnow().isoformat() + "Z")
    store.setdefault("logs", []).append(entry)
    _save_store(store)
    return entry


def list_logs(limit: int = 200) -> List[Dict[str, Any]]:
    # Prefer Supabase if configured
    if _sb:
        try:
            res = _sb.table("audit_logs").select("*")\
                .order("created_at", desc=True).limit(limit).execute()
            out: List[Dict[str, Any]] = []
            if hasattr(res, "data"):
                for row in res.data:
                    row = dict(row)
                    row["time"] = row.get("created_at")
                    out.append(row)
            return out
        except Exception:
            pass
    store = _ensure_store()
    logs = store.get("logs", [])
    return logs[-limit:][::-1]


