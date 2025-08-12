#!/usr/bin/env python3
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "milvus_exports"
    if not out_dir.exists():
        print(f"No export directory found at {out_dir}")
        return

    # Gather all JSON files deterministically
    json_paths = sorted(out_dir.glob("*.json"))
    next_id = 0

    for path in json_paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"Skipping non-list JSON: {path}")
                continue

            for rec in data:
                rec["chunk_id"] = next_id
                next_id += 1

            tmp_path = path.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp_path.replace(path)
            print(f"Rewrote {path.name}")
        except Exception as e:
            print(f"Failed to rewrite {path}: {e}")

    print(f"Done. Assigned chunk_id 0..{next_id-1}")


if __name__ == "__main__":
    main()


