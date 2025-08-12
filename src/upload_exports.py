#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List, Dict, Any
import requests

from constants import MILVUS_URI, MILVUS_TOKEN, MILVUS_DATABASE
from milvus_connector import ensure_collection_exists


INSERT_URL = f"{MILVUS_URI}/v2/vectordb/entities/insert"
HEADERS = {
    "Authorization": f"Bearer {MILVUS_TOKEN}",
    "Content-Type": "application/json",
}
BATCH_SIZE = 50


def upload_file(json_path: Path) -> int:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or "data" not in payload:
        print(f"Skipping {json_path.name}: not in expected object format")
        return 0

    collection_name = payload.get("collectionName")
    if not collection_name:
        print(f"Skipping {json_path.name}: missing collectionName")
        return 0

    records: List[Dict[str, Any]] = payload["data"]
    if not isinstance(records, list) or not records:
        print(f"Skipping {json_path.name}: empty data array")
        return 0

    total_uploaded = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        body = {
            "dbName": MILVUS_DATABASE,
            "collectionName": collection_name,
            "data": batch,
        }
        resp = requests.post(INSERT_URL, headers=HEADERS, json=body)
        if resp.status_code != 200:
            print(f"Insert failed for {json_path.name} [{i}-{i+len(batch)-1}]: {resp.status_code} {resp.text[:300]}")
            continue
        total_uploaded += len(batch)

    return total_uploaded


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exports_dir = root / "milvus_exports"
    if not exports_dir.exists():
        print(f"No export directory found at {exports_dir}")
        return

    # Ensure collection exists (safe if already present)
    try:
        ensure_collection_exists()
    except Exception:
        pass

    json_files = sorted(exports_dir.glob("*.json"))
    if not json_files:
        print("No JSON files to upload.")
        return

    grand_total = 0
    for jf in json_files:
        uploaded = upload_file(jf)
        print(f"Uploaded {uploaded} records from {jf.name}")
        grand_total += uploaded

    print(f"Done. Uploaded {grand_total} records in total.")


if __name__ == "__main__":
    main()


