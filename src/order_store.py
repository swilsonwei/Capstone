import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STORE_PATH = DATA_DIR / "orders.json"

# Optional: Supabase persistence
try:
    from supabase import create_client
    from constants import SUPABASE_URL, SUPABASE_SERVICE_KEY
    _sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_KEY else None
except Exception:
    _sb = None


def _ensure_store() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        store = {"last_id": 0, "orders": []}
        STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")
        return store
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"last_id": 0, "orders": []}


def _save_store(store: Dict[str, Any]) -> None:
    STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _format_order_id(n: int) -> str:
    return f"OD-{n:05d}"


def create_order(source_file: str, subtotal: float, items_count: int, status: str = "Quoted") -> Dict[str, Any]:
    store = _ensure_store()
    store["last_id"] = int(store.get("last_id", 0)) + 1
    order_id = _format_order_id(store["last_id"])
    order = {
        "id": order_id,
        "source_file": source_file,
        "items_count": int(items_count or 0),
        "subtotal": float(subtotal or 0.0),
        "status": status,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    store.setdefault("orders", []).append(order)
    _save_store(store)

    # Also upsert into Supabase if configured
    if _sb:
        try:
            _sb.table("orders").upsert(order).execute()
        except Exception:
            pass
    return order


def list_orders() -> List[Dict[str, Any]]:
    # Prefer Supabase if configured; fallback to local store
    if _sb:
        try:
            resp = _sb.table("orders").select("*").order("created_at", desc=True).execute()
            if resp and hasattr(resp, "data"):
                return list(resp.data)
        except Exception:
            pass
    store = _ensure_store()
    return list(store.get("orders", []))


def update_status(order_id: str, status: str) -> Dict[str, Any]:
    store = _ensure_store()
    for o in store.get("orders", []):
        if o.get("id") == order_id:
            o["status"] = status
            _save_store(store)
            if _sb:
                try:
                    _sb.table("orders").update({"status": status}).eq("id", order_id).execute()
                except Exception:
                    pass
            return o
    return {}


