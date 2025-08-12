#!/usr/bin/env python3
import json
from pathlib import Path
from constants import COLLECTION_NAME


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "milvus_exports"
    if not out_dir.exists():
        print(f"No export directory found at {out_dir}")
        return

    for path in sorted(out_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "data" in data and "collectionName" in data:
                print(f"Already wrapped: {path.name}")
                continue
            if not isinstance(data, list):
                print(f"Skipping non-list JSON: {path.name}")
                continue
            wrapped = {"collectionName": COLLECTION_NAME, "data": data}
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(wrapped, f, ensure_ascii=False, indent=2)
            tmp.replace(path)
            print(f"Wrapped: {path.name}")
        except Exception as e:
            print(f"Failed to wrap {path.name}: {e}")


if __name__ == "__main__":
    main()


