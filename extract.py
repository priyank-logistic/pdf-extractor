#!/usr/bin/env python3
"""Direct PDF page images -> one vision-model request -> normalized JSON."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

try:
    from PIL import Image
except ImportError:  # pragma: no cover - environment-dependent diagnostic
    Image = None


DEFAULT_URL = "http://192.168.0.254:11434"
DEFAULT_MODEL = "gemma4:12b"
SCRIPT_DIR = Path(__file__).resolve().parent

FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "value": {"type": ["string", "null"]},
    },
    "required": ["label", "value"],
    "additionalProperties": False,
}

ROW_SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {"type": "array", "items": FIELD_SCHEMA},
        "source_page": {"type": "integer", "minimum": 1},
    },
    "required": ["cells", "source_page"],
    "additionalProperties": False,
}

MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "document_title": {"type": ["string", "null"]},
        "metadata": {"type": "array", "items": FIELD_SCHEMA},
        "headers": {"type": "array", "items": {"type": "string"}},
        "rows": {"type": "array", "items": ROW_SCHEMA},
        "notes": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["document_title", "metadata", "headers", "rows", "notes", "warnings"],
    "additionalProperties": False,
}

PROMPT = """Read every supplied page of this price-list PDF and return one complete flat row table as JSON.
For readability, each original page may be supplied as overlapping top-to-bottom panels. The panel manifest
below gives the original page number for every image. Rejoin the panels mentally and do not duplicate rows
visible in an overlap. Perform the visual reading and hierarchy analysis yourself.

Completeness is the highest priority:
- Find every physical body row containing a product/collection/design/quality and a price or genuine table value.
- Return exactly one `rows` item per physical table row. Do not nest or group rows.
- Represent every physical row exactly once. Do not omit or duplicate rows.
- Before answering, visually scan every page from top to bottom and audit that all price rows appear once.
- Copy spelling, punctuation, decimal zeroes, currency symbols, percentages and dashes exactly as printed. Never calculate values.

Rows and merged cells:
- Expand vertically merged cells: repeat their value in every physical row covered by that cell.
- Example: if SALISTER spans two rows beside 6388 and 6399, return two flat rows and repeat SALISTER in both.
- Likewise repeat a merged serial number or collection value such as NO. 2 / AADHYA in both KLOTH and SHIAR rows.
- Use visible borders, merged-cell extents and alignment to decide which physical rows a value covers.
- Do not return a `qualities` array and do not create a catalog hierarchy.

Headers and cells:
- Do not map printed headers to predefined keys. Every PDF can use different headers.
- Return one leaf-column label in `headers` for every physical table column, in left-to-right order.
- Preserve each visibly printed leaf header exactly, including case and spelling.
- If one printed spanning header covers two physical columns and the second leaf column has no separately
  printed header, keep the printed header for the first leaf column and use exactly `quality` for the unnamed
  second leaf column. This is the only synthesized header.
- Do not use `quality` when the second column has its own printed header such as DESIGN-SHADE; preserve that header.
- Each row returns `cells` as `{\"label\", \"value\"}` pairs. Use the exact corresponding header as `label`.
- Include blank cells with a null value when possible. A genuinely omitted blank cell will be restored as null.

Exclude titles, addresses, contact details, logos, page numbers, terms and notes from catalogs.
Use 1-based `source_pages`. Omit optional fields that are not printed; do not invent values.
Return JSON only and exactly match the supplied schema."""


class DirectExtractionError(RuntimeError):
    pass


def render_pages(pdf: Path, destination: Path, dpi: int) -> list[Path]:
    if shutil.which("pdftoppm") is None:
        raise DirectExtractionError("Required command is not installed: pdftoppm")
    prefix = destination / "page"
    process = subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(prefix)],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise DirectExtractionError(process.stderr.strip() or "Could not render PDF")
    pages = sorted(destination.glob("page-*.png"), key=_page_number)
    if not pages:
        raise DirectExtractionError("PDF renderer produced no page images")
    return pages


def _page_number(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    return int(match.group(1)) if match else 0


def make_panels(pages: list[Path], destination: Path, panel_count: int) -> list[tuple[int, Path]]:
    if panel_count <= 1:
        return [(index, page) for index, page in enumerate(pages, 1)]
    if Image is None:
        raise DirectExtractionError("Pillow is required for --panels greater than 1")
    result: list[tuple[int, Path]] = []
    for page_number, page in enumerate(pages, 1):
        with Image.open(page) as source:
            width, height = source.size
            base_height = height / panel_count
            overlap = int(base_height * 0.12)
            for panel_number in range(panel_count):
                top = max(0, int(panel_number * base_height) - (overlap if panel_number else 0))
                bottom = min(height, int((panel_number + 1) * base_height) +
                             (overlap if panel_number + 1 < panel_count else 0))
                target = destination / f"page-{page_number}-panel-{panel_number + 1}.png"
                source.crop((0, top, width, bottom)).save(target, "PNG")
                result.append((page_number, target))
    return result


def _request_model(
    inputs: list[tuple[int, Path]], url: str, model: str, timeout: float
) -> tuple[dict, dict]:
    images = [base64.b64encode(path.read_bytes()).decode("ascii") for _, path in inputs]
    manifest = "\n".join(
        f"Image {index}: original PDF page {page}, panel {path.stem.rsplit('-', 1)[-1]}"
        for index, (page, path) in enumerate(inputs, 1)
    )
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 65536, "num_predict": 20000},
        "format": MODEL_SCHEMA,
        "messages": [{"role": "user", "content": PROMPT + "\n\nPANEL MANIFEST:\n" + manifest,
                      "images": images}],
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            envelope = json.load(response)
        content = envelope["message"]["content"]
        result = json.loads(content)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise DirectExtractionError(f"Ollama HTTP {exc.code}: {detail or exc.reason}") from exc
    except (OSError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        raise DirectExtractionError(f"Model request failed: {exc}") from exc
    if not isinstance(result, dict) or not isinstance(result.get("rows"), list):
        raise DirectExtractionError("Model response does not contain a rows array")
    original_pages = sorted({page for page, _ in inputs})
    if len(original_pages) == 1:
        for row in result["rows"]:
            if isinstance(row, dict):
                row["source_page"] = original_pages[0]
    normalize_dynamic_labels(result)
    validate_header_labels(result)
    return result, envelope


def call_model(
    inputs: list[tuple[int, Path]],
    url: str,
    model: str,
    timeout: float,
    retries: int = 1,
) -> dict:
    errors: list[str] = []
    for attempt in range(retries + 1):
        try:
            result, _ = _request_model(inputs, url, model, timeout)
            return result
        except DirectExtractionError as exc:
            errors.append(f"attempt {attempt + 1}: {exc}")
    raise DirectExtractionError("Model request failed after retries: " + "; ".join(errors))


def validate_header_labels(result: dict) -> None:
    """Require labels to be known and unambiguous; missing blank cells are allowed."""
    headers = result.get("headers")
    if not isinstance(headers, list) or not headers or any(not isinstance(item, str) for item in headers):
        raise DirectExtractionError("Model response does not contain a valid headers array")
    for row_number, row in enumerate(result.get("rows", []), 1):
        cells = row.get("cells") if isinstance(row, dict) else None
        if not isinstance(cells, list):
            raise DirectExtractionError(f"Model row {row_number} has no cells array")
        labels = [cell.get("label") for cell in cells if isinstance(cell, dict)]
        unknown = [label for label in labels if label not in headers]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if unknown:
            raise DirectExtractionError(
                f"Model row {row_number} contains labels absent from headers: {unknown}"
            )
        if duplicates:
            raise DirectExtractionError(
                f"Model row {row_number} contains duplicate labels: {duplicates}"
            )


def normalize_dynamic_labels(result: dict) -> None:
    """Make blank/duplicate model headers usable without imposing vendor-specific fields."""
    raw_headers = result.get("headers")
    if not isinstance(raw_headers, list):
        return
    normalized: list[str] = []
    occurrences: dict[str, int] = {}
    header_variants: dict[str, list[str]] = {}
    for index, raw_header in enumerate(raw_headers, 1):
        raw = str(raw_header or "").strip()
        base = raw or f"column_{index}"
        occurrences[base] = occurrences.get(base, 0) + 1
        header = base if occurrences[base] == 1 else f"{base}_{occurrences[base]}"
        normalized.append(header)
        header_variants.setdefault(raw, []).append(header)
    result["headers"] = normalized

    for row in result.get("rows", []):
        cells = row.get("cells", []) if isinstance(row, dict) else []
        used: set[str] = set()
        label_occurrences: dict[str, int] = {}
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            raw = str(cell.get("label") or "").strip()
            label_occurrences[raw] = label_occurrences.get(raw, 0) + 1
            variants = header_variants.get(raw, [])
            variant_index = label_occurrences[raw] - 1
            if variant_index < len(variants):
                label = variants[variant_index]
            elif index < len(normalized):
                label = normalized[index]
            else:
                base = raw or f"column_{index + 1}"
                label = base
                suffix = 2
                while label in result["headers"]:
                    label = f"{base}_{suffix}"
                    suffix += 1
                result["headers"].append(label)
            if label in used:
                replacement = next((item for item in normalized if item not in used), None)
                if replacement:
                    label = replacement
            cell["label"] = label
            used.add(label)


def _flat_rows(model_result: dict) -> list[dict]:
    source_rows = model_result.get("rows", [])
    rows = []
    for source_row in source_rows:
        values_by_label = {cell["label"]: cell.get("value") for cell in source_row["cells"]}
        record = {header: values_by_label.get(header) for header in model_result["headers"]}
        record["source_page"] = source_row["source_page"]
        rows.append(record)
    return rows


def merge_model_results(pdf: Path, page_count: int, model_results: list[dict]) -> dict:
    """Preserve genuinely different page headers as independent dynamic tables."""
    tables: list[dict] = []
    row_count = 0
    metadata: list[dict] = []
    notes: list[str] = []
    warning_messages: list[str] = []
    document_title = next(
        (item.get("document_title") for item in model_results if item.get("document_title")), None
    )

    for model_result in model_results:
        headers = model_result.get("headers", [])
        rows = _flat_rows(model_result)
        pages = sorted({row["source_page"] for row in rows})
        table = next((item for item in tables if item["headers"] == headers), None)
        if table is None:
            table = {"table": len(tables) + 1, "headers": headers, "source_pages": [], "rows": []}
            tables.append(table)
        table["source_pages"] = sorted(set(table["source_pages"] + pages))
        table["rows"].extend(rows)
        row_count += len(rows)
        for field in model_result.get("metadata", []):
            if field not in metadata:
                metadata.append(field)
        for note in model_result.get("notes", []):
            if note not in notes:
                notes.append(note)
        warning_messages.extend(str(item) for item in model_result.get("warnings", []))

    return {
        "schema_version": "3.0",
        "source": {"file_name": pdf.name, "page_count": page_count},
        "document_type": "catalog_price_list",
        "data": {
            "document_title": document_title,
            "metadata": metadata,
            "tables": tables,
            "notes": notes,
        },
        "warnings": [
            {"code": "MODEL_WARNING", "message": message,
             "page": None, "table": None, "row": None}
            for message in warning_messages
        ],
        "metrics": {
            "tables_found": len(tables),
            "rows_extracted": row_count,
            "rows_with_warnings": len(warning_messages),
            "extraction_methods": ["direct_document_vision"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send all PDF page images directly to one vision model")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", DEFAULT_URL))
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("OLLAMA_TIMEOUT", "600")))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--panels", type=int, default=3,
                        help="overlapping vertical panels per page; use 1 for whole pages")
    parser.add_argument("--pages-per-request", type=int, default=1,
                        help="original PDF pages per model request; default 1 prevents truncation")
    parser.add_argument("--model-retries", type=int, default=1,
                        help="retries after malformed JSON or request failure")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf = args.input
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise DirectExtractionError(f"Not a readable PDF: {pdf}")
    # Keep the default independent of both the current working directory and
    # the directory name containing this standalone script.
    output = args.output or SCRIPT_DIR / "output" / f"{pdf.stem}.json"
    if output.exists() and not args.overwrite:
        raise DirectExtractionError(f"Output exists; use --overwrite: {output}")
    with tempfile.TemporaryDirectory(prefix="direct-pdf-model-") as temporary:
        temporary_path = Path(temporary)
        pages = render_pages(pdf, temporary_path, args.dpi)
        inputs = make_panels(pages, temporary_path, max(1, args.panels))
        pages_per_request = max(1, args.pages_per_request)
        model_results: list[dict] = []
        for start in range(1, len(pages) + 1, pages_per_request):
            end = min(len(pages), start + pages_per_request - 1)
            request_inputs = [item for item in inputs if start <= item[0] <= end]
            print(f"Extracting PDF page(s) {start}-{end}...", flush=True)
            model_results.append(call_model(
                request_inputs,
                args.ollama_url,
                args.ollama_model,
                args.timeout,
                max(0, args.model_retries),
            ))
        result = merge_model_results(pdf, len(pages), model_results)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    temporary_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_output.replace(output)
    print(f"{pdf} -> {output} ({result['metrics']['rows_extracted']} rows, "
          f"{len(result['warnings'])} warnings)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DirectExtractionError as exc:
        raise SystemExit(f"ERROR: {exc}")
