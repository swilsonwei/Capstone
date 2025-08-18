import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STORE_PATH = DATA_DIR / "orders.json"

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


def _parse_order_id(order_id: str) -> int:
    try:
        return int(str(order_id).split("-")[-1])
    except Exception:
        return 0


def _next_id_from_supabase() -> int:
    """Get the next sequential numeric id using Supabase as source of truth."""
    if not _sb:
        return 0
    try:
        # IDs are zero-padded, so lexicographic desc works
        resp = _sb.table("orders").select("id").order("id", desc=True).limit(1).execute()
        last = (resp.data or [])[0]["id"] if hasattr(resp, "data") and resp.data else None
        last_n = _parse_order_id(last) if last else 0
        return last_n + 1
    except Exception:
        return 0


def create_order(source_file: str, subtotal: float, items_count: int, status: str = "Quoted", items: List[Dict[str, Any]] | None = None, prompt: Optional[str] = None, tool_name: Optional[str] = None) -> Dict[str, Any]:
    store = _ensure_store()
    # Determine next id. Prefer Supabase (shared, persistent); fallback to local JSON counter.
    next_n = _next_id_from_supabase() if _sb else 0
    if next_n <= 0:
        store["last_id"] = int(store.get("last_id", 0)) + 1
        next_n = store["last_id"]
    order_id = _format_order_id(next_n)
    order = {
        "id": order_id,
        "source_file": source_file,
        "customers": "",
        "items_count": int(items_count or 0),
        "subtotal": float(subtotal or 0.0),
        "status": status,
        "created_at": datetime.utcnow().isoformat() + "Z",
        # store items locally for PDF generation (not pushed to Supabase)
        "items": items or [],
    }
    # Keep local store in sync (does not replace Supabase persistence)
    # Ensure local last_id is at least next_n
    store["last_id"] = max(int(store.get("last_id", 0)), next_n)
    store.setdefault("orders", []).append(order)
    _save_store(store)

    # Also upsert into Supabase if configured
    if _sb:
        try:
            # Try inserting with items (JSONB column if present)
            _sb.table("orders").insert(order).execute()
        except Exception:
            # Fallback: insert without items field if column doesn't exist
            try:
                row_no_items = {k: v for k, v in order.items() if k != "items"}
                _sb.table("orders").insert(row_no_items).execute()
            except Exception:
                pass
    # Emit audit log (best-effort)
    try:
        from src.audit_log import append_log
        append_log({
            "type": "order_created",
            "order_id": order_id,
            "details": {"source_file": source_file, "items_count": items_count, "subtotal": subtotal},
            "prompt": prompt,
            "tool_name": tool_name,
        })
    except Exception:
        pass
    return order


def list_orders() -> List[Dict[str, Any]]:
    # Prefer Supabase if configured; fallback to local store
    if _sb:
        try:
            resp = _sb.table("orders").select("*").order("created_at", desc=True).execute()
            if resp and hasattr(resp, "data"):
                sb_orders = [dict(r) for r in list(resp.data)]
                # Merge local cache fields (e.g., customers, items) if Supabase rows are missing them
                store = _ensure_store()
                local_by_id = {o.get("id"): o for o in store.get("orders", [])}
                merged: List[Dict[str, Any]] = []
                for row in sb_orders:
                    oid = row.get("id")
                    local = local_by_id.get(oid, {})
                    # Prefer Supabase values; fill blanks from local cache
                    if (not row.get("customers")) and local.get("customers"):
                        row["customers"] = local.get("customers")
                    # Do not override Supabase items; only fill if absent
                    if (not row.get("items")) and local.get("items"):
                        row["items"] = local.get("items")
                    merged.append(row)
                return merged
        except Exception:
            pass
    store = _ensure_store()
    return list(store.get("orders", []))


def update_status(order_id: str, status: str, prompt: str | None = None, tool_name: Optional[str] = None) -> Dict[str, Any]:
    # If Supabase is configured, check current status and update only if it changed
    if _sb:
        try:
            # Read current status first
            curr_resp = _sb.table("orders").select("id,status,source_file,items_count,subtotal,created_at,items").eq("id", order_id).limit(1).execute()
            curr_row = (curr_resp.data or [])[0] if hasattr(curr_resp, "data") and curr_resp.data else None
            if curr_row and str(curr_row.get("status")) == str(status):
                # No change; ensure local cache mirrors and return without logging
                store = _ensure_store()
                found = None
                for o in store.get("orders", []):
                    if o.get("id") == order_id:
                        o["status"] = status
                        found = o
                        break
                if found is None:
                    found = {
                        "id": curr_row.get("id"),
                        "source_file": curr_row.get("source_file", ""),
                        "items_count": int(curr_row.get("items_count", 0) or 0),
                        "subtotal": float(curr_row.get("subtotal", 0) or 0.0),
                        "status": status,
                        "created_at": curr_row.get("created_at") or datetime.utcnow().isoformat() + "Z",
                        "items": curr_row.get("items") or [],
                    }
                    store.setdefault("orders", []).append(found)
                _save_store(store)
                return found

            _sb.table("orders").update({"status": status}).eq("id", order_id).execute()
            resp = _sb.table("orders").select("*").eq("id", order_id).limit(1).execute()
            sb_order = (resp.data or [])[0] if hasattr(resp, "data") and resp.data else None
            if sb_order:
                # Sync local cache if present; if missing, create a minimal local record
                store = _ensure_store()
                target = None
                for o in store.get("orders", []):
                    if o.get("id") == order_id:
                        o["status"] = status
                        target = o
                        break
                if target is None:
                    target = {
                        "id": sb_order.get("id"),
                        "source_file": sb_order.get("source_file", ""),
                        "items_count": int(sb_order.get("items_count", 0) or 0),
                        "subtotal": float(sb_order.get("subtotal", 0) or 0.0),
                        "status": status,
                        "created_at": sb_order.get("created_at") or datetime.utcnow().isoformat() + "Z",
                        "items": [],
                    }
                    store.setdefault("orders", []).append(target)
                _save_store(store)
                # Single-source audit log (data layer only)
                try:
                    from src.audit_log import append_log
                    append_log({
                        "type": "order_status_updated",
                        "order_id": order_id,
                        "details": {"status": status},
                        "prompt": prompt,
                        "tool_name": tool_name,
                    })
                except Exception:
                    pass
                return target
        except Exception:
            # fall back to local-only update below
            pass

    # Local-only fallback
    store = _ensure_store()
    for o in store.get("orders", []):
        if o.get("id") == order_id:
            o["status"] = status
            _save_store(store)
            # Single-source audit log (data layer only)
            try:
                from src.audit_log import append_log
                append_log({
                    "type": "order_status_updated",
                    "order_id": order_id,
                    "details": {"status": status},
                    "prompt": prompt,
                    "tool_name": tool_name,
                })
            except Exception:
                pass
            return o
    return {}


def get_order(order_id: str) -> Dict[str, Any] | None:
    store = _ensure_store()
    local: Dict[str, Any] | None = None
    for o in store.get("orders", []):
        if o.get("id") == order_id:
            local = o
            break
    # Prefer Supabase when available; merge into local cache to keep items in sync
    if _sb:
        try:
            resp = _sb.table("orders").select("*").eq("id", order_id).limit(1).execute()
            data = (resp.data or []) if hasattr(resp, "data") else []
            sb_row = dict(data[0]) if data else None
        except Exception:
            sb_row = None
        if sb_row is not None:
            # If Supabase has items and local does not, propagate to local cache
            if sb_row.get("items") and local and not local.get("items"):
                try:
                    local["items"] = sb_row.get("items")
                    local["items_count"] = int(sb_row.get("items_count", len(local.get("items", [])) or 0))
                    local["subtotal"] = float(sb_row.get("subtotal", local.get("subtotal", 0)))
                    _save_store(store)
                except Exception:
                    pass
            # If Supabase lacks customers but local has it, hydrate return value
            try:
                if (not sb_row.get("customers")) and local and local.get("customers"):
                    sb_row["customers"] = local.get("customers")
            except Exception:
                pass
            # If local has items but Supabase lacks them, enrich the return value with local items
            if (not sb_row.get("items")) and local and local.get("items"):
                sb_row["items"] = local.get("items")
            return sb_row
    # Fallback to local if present
    return local


def update_items(order_id: str, items: List[Dict[str, Any]], prompt: Optional[str] = None, tool_name: Optional[str] = None) -> Dict[str, Any] | None:
    """Replace line items for the order; recompute subtotal and item count."""
    store = _ensure_store()
    target = None
    for o in store.get("orders", []):
        if o.get("id") == order_id:
            target = o
            break
    # If not in local cache but Supabase has it, seed local record
    if target is None and _sb:
        try:
            resp = _sb.table("orders").select("*").eq("id", order_id).limit(1).execute()
            sb_row = (resp.data or [])[0] if hasattr(resp, "data") and resp.data else None
            if sb_row:
                target = {
                    "id": sb_row.get("id"),
                    "source_file": sb_row.get("source_file", ""),
                    "customers": sb_row.get("customers", ""),
                    "items": list(sb_row.get("items") or []),
                    "items_count": int(sb_row.get("items_count", 0) or 0),
                    "subtotal": float(sb_row.get("subtotal", 0) or 0.0),
                    "status": sb_row.get("status", "Quoted"),
                    "created_at": sb_row.get("created_at") or datetime.utcnow().isoformat() + "Z",
                }
                store.setdefault("orders", []).append(target)
                _save_store(store)
        except Exception:
            pass
    if not target:
        return None
    # Guardrail: block edits for Sent/Received
    try:
        if str(target.get("status")) in ("Sent", "Received"):
            try:
                from src.audit_log import append_log
                append_log({
                    "type": "order_items_update_blocked",
                    "order_id": order_id,
                    "details": {"reason": "read-only status", "status": target.get("status")},
                    "prompt": prompt,
                    "tool_name": tool_name,
                })
            except Exception:
                pass
            return None
    except Exception:
        pass
    # normalize numbers and compute subtotal
    normalized = []
    subtotal = 0.0
    for it in items or []:
        name = str(it.get("item", "")).strip()
        try:
            qty = float(it.get("quantity", 0))
        except Exception:
            qty = 0.0
        try:
            unit_cost = float(it.get("unit_cost", it.get("cost", 0)))
        except Exception:
            unit_cost = 0.0
        line_total = qty * unit_cost
        subtotal += line_total
        normalized.append({
            "item": name,
            "quantity": qty,
            "unit_cost": unit_cost,
            "line_total": line_total,
        })
    target["items"] = normalized
    target["items_count"] = len(normalized)
    target["subtotal"] = float(subtotal)
    _save_store(store)

    # update Supabase aggregates
    if _sb:
        try:
            # Attempt to persist items JSON if column exists
            _sb.table("orders").update({
                "items": normalized,
                "items_count": len(normalized),
                "subtotal": float(subtotal)
            }).eq("id", order_id).execute()
        except Exception:
            # Fallback without items column
            try:
                _sb.table("orders").update({
                    "items_count": len(normalized),
                    "subtotal": float(subtotal)
                }).eq("id", order_id).execute()
            except Exception:
                pass
    # Audit log
    try:
        from src.audit_log import append_log
        append_log({
            "type": "order_items_updated",
            "order_id": order_id,
            "details": {"items_count": len(normalized), "subtotal": float(subtotal)},
            "prompt": prompt,
            "tool_name": tool_name,
        })
    except Exception:
        pass
    return target


def update_customer(order_id: str, customers: str, prompt: Optional[str] = None, tool_name: Optional[str] = None) -> Dict[str, Any] | None:
    """Update the customer name for an order (both Supabase and local cache)."""
    def _normalize_customer_name(name: str) -> str:
        try:
            s = (name or "").strip()
            if not s:
                return ""
            # Preserve acronyms that arrive uppercase (e.g., BMS, GSK)
            if s.isupper():
                return s
            # If clearly all-lower, convert to Title Case for readability
            if s.islower():
                # Title-case alphabetic runs while preserving numerics/symbols
                return re.sub(r"[A-Za-z]+", lambda m: m.group(0).capitalize(), s)
            return s
        except Exception:
            return (name or "").strip()

    customers = _normalize_customer_name(customers)
    # Supabase-first
    if _sb:
        try:
            _sb.table("orders").update({"customers": customers}).eq("id", order_id).execute()
            resp = _sb.table("orders").select("*").eq("id", order_id).limit(1).execute()
            sb_order = (resp.data or [])[0] if hasattr(resp, "data") and resp.data else None
            if sb_order:
                # Sync local cache
                store = _ensure_store()
                target = None
                for o in store.get("orders", []):
                    if o.get("id") == order_id:
                        o["customers"] = customers
                        target = o
                        break
                if target is None:
                    target = {
                        "id": sb_order.get("id"),
                        "source_file": sb_order.get("source_file", ""),
                        "customers": customers,
                        "items_count": int(sb_order.get("items_count", 0) or 0),
                        "subtotal": float(sb_order.get("subtotal", 0) or 0.0),
                        "status": sb_order.get("status", "Quoted"),
                        "created_at": sb_order.get("created_at") or datetime.utcnow().isoformat() + "Z",
                        "items": sb_order.get("items") or [],
                    }
                    store.setdefault("orders", []).append(target)
                _save_store(store)
                try:
                    from src.audit_log import append_log
                    append_log({
                        "type": "order_customer_updated",
                        "order_id": order_id,
                        "details": {"customers": customers},
                        "prompt": prompt,
                        "tool_name": tool_name,
                    })
                except Exception:
                    pass
                return target
        except Exception:
            # Fall through to local handling
            pass

    # Ensure local cache has a record for this order by hydrating from Supabase or get_order
    store = _ensure_store()
    target = None
    for o in store.get("orders", []):
        if o.get("id") == order_id:
            target = o
            break
    if target is None:
        # Try to hydrate from Supabase directly
        if _sb:
            try:
                resp = _sb.table("orders").select("*").eq("id", order_id).limit(1).execute()
                sb_row = (resp.data or [])[0] if hasattr(resp, "data") and resp.data else None
            except Exception:
                sb_row = None
            if sb_row:
                target = {
                    "id": sb_row.get("id"),
                    "source_file": sb_row.get("source_file", ""),
                    "customers": customers,
                    "items_count": int(sb_row.get("items_count", 0) or 0),
                    "subtotal": float(sb_row.get("subtotal", 0) or 0.0),
                    "status": sb_row.get("status", "Quoted"),
                    "created_at": sb_row.get("created_at") or datetime.utcnow().isoformat() + "Z",
                    "items": sb_row.get("items") or [],
                }
                store.setdefault("orders", []).append(target)
                _save_store(store)
    if target is None:
        # As a last resort, fetch via get_order (may merge supabase/local)
        try:
            merged = get_order(order_id)
            if merged:
                target = dict(merged)
                store.setdefault("orders", []).append(target)
                _save_store(store)
        except Exception:
            pass

    if target is not None:
        target["customers"] = customers
        _save_store(store)
        try:
            from src.audit_log import append_log
            append_log({
                "type": "order_customer_updated",
                "order_id": order_id,
                "details": {"customers": customers},
                "prompt": prompt,
                "tool_name": tool_name,
            })
        except Exception:
            pass
        return target

    return None


def delete_order(order_id: str) -> bool:
    """Delete an order from local store and Supabase (if configured) and log an audit entry."""
    store = _ensure_store()
    before = len(store.get("orders", []))
    store["orders"] = [o for o in store.get("orders", []) if o.get("id") != order_id]
    after = len(store["orders"])
    changed = after < before
    if changed:
        _save_store(store)
    if _sb:
        try:
            _sb.table("orders").delete().eq("id", order_id).execute()
        except Exception:
            pass
    if changed:
        try:
            from src.audit_log import append_log
            append_log({"type": "order_deleted", "order_id": order_id})
        except Exception:
            pass
    return changed



