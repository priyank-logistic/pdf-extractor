#!/usr/bin/env python3
"""Migrate direct-extractor JSON files to the non-duplicated tables-only shape."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def migrate(path: Path) -> bool:
    document = json.loads(path.read_text(encoding="utf-8"))
    data = document.get("data")
    if not isinstance(data, dict):
        return False

    tables = data.get("tables")
    changed = False
    if not isinstance(tables, list) or not tables:
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        pages = sorted({row.get("source_page") for row in rows if row.get("source_page") is not None})
        data["tables"] = [{
            "table": 1,
            "headers": headers,
            "source_pages": pages,
            "rows": rows,
        }]
        changed = True

    if "headers" in data:
        del data["headers"]
        changed = True
    if "rows" in data:
        del data["rows"]
        changed = True

    tables = data.get("tables", [])
    metrics = document.setdefault("metrics", {})
    metrics["tables_found"] = len(tables)
    metrics["rows_extracted"] = sum(len(table.get("rows", [])) for table in tables)

    if changed:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("direct_model_extractor/output"))
    args = parser.parse_args()
    changed = 0
    for path in sorted(args.directory.glob("*.json")):
        if migrate(path):
            changed += 1
            print(f"migrated: {path}")
    print(f"Migration complete: {changed} file(s) changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
