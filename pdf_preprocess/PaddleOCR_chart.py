from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from paddlex import create_model


def _as_dict(x: Any) -> Optional[Dict[str, Any]]:
    """Best-effort: convert various PaddleX result representations to a dict."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x

    # PaddleX result objects often provide `json` attribute (dict) or `to_json()` (str).
    j = getattr(x, "json", None)
    if isinstance(j, dict):
        return j
    if isinstance(j, str):
        try:
            obj = json.loads(j)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    to_json = getattr(x, "to_json", None)
    if callable(to_json):
        try:
            obj = json.loads(to_json())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Fallback: inspect __dict__
    d = getattr(x, "__dict__", None)
    if isinstance(d, dict):
        return d

    return None


def extract_table_text(result_obj: Any) -> Optional[str]:
    """
    Extract the table string from PaddleX PP-Chart2Table output.

    According to the official example output, the prediction can look like:
      {'res': {'image': ..., 'result': 'col1 | col2 ...\\nrow1 | ...'}}
    """
    d = _as_dict(result_obj)
    if not isinstance(d, dict):
        return None

    inner = d.get("res", d)
    if not isinstance(inner, dict):
        return None

    text = inner.get("result")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return None


def _split_pipe_row(line: str) -> List[str]:
    # Handle both:
    #   "a | b | c"
    # and markdown-ish:
    #   "| a | b | c |"
    s = line.strip()
    s = s.strip("|").strip()
    if not s:
        return []
    return [c.strip() for c in s.split("|")]


def _is_separator_row(cells: List[str]) -> bool:
    # Markdown separator: --- / :---: / ---:
    def is_sep_cell(cell: str) -> bool:
        t = cell.strip()
        if not t:
            return False
        return all(ch in "-: " for ch in t) and t.strip("-: ") == ""
    return bool(cells) and all(is_sep_cell(c) for c in cells)


def parse_table_to_rows(table_text: str) -> List[List[str]]:
    """
    Parse the table text into rows.

    PP-Chart2Table commonly returns pipe-separated text (not necessarily full markdown).
    We:
      - take first pipe-row as header
      - skip markdown separator rows (---)
      - try to merge wrapped lines (rare) by buffering until it matches header column count
    """
    lines = [ln.strip() for ln in table_text.splitlines() if ln.strip()]
    if not lines:
        return []

    # Find the first line that looks like a pipe-separated row.
    header_cells: List[str] = []
    start_idx = 0
    for i, ln in enumerate(lines):
        cells = _split_pipe_row(ln)
        if len(cells) >= 2:  # a table row should have at least 2 columns
            header_cells = cells
            start_idx = i + 1
            break

    if not header_cells:
        # Fallback: single-column text
        return [[table_text.strip()]]

    ncols = len(header_cells)
    rows: List[List[str]] = [header_cells]

    buf: List[str] = []
    for ln in lines[start_idx:]:
        cells = _split_pipe_row(ln)
        if cells and _is_separator_row(cells):
            continue

        candidate = " ".join(buf + [ln]) if buf else ln
        cand_cells = _split_pipe_row(candidate)

        if len(cand_cells) == ncols:
            rows.append(cand_cells)
            buf = []
        else:
            # If a row label wraps to next line, buffer and retry.
            buf.append(ln)

    # If leftover buffer exists, append as a single cell row to avoid data loss
    if buf:
        rows.append([" ".join(buf)] + [""] * (ncols - 1))

    # Normalize rectangle
    return [r + [""] * (ncols - len(r)) if len(r) < ncols else r[:ncols] for r in rows]


def write_csv(rows: List[List[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def write_markdown(rows: List[List[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    header = rows[0]
    sep = ["---"] * len(header)

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows[1:]:
        r = (r + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(r) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    in_dir = Path("../Data/Charts")
    out_dir = Path("./output_charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = create_model("PP-Chart2Table")

    for img_path in sorted(in_dir.glob("*.png")):
        results = model.predict(input={"image": str(img_path)}, batch_size=1)
        for res in results:
            # visibility/debug (PaddleX result objects support .print())
            try:
                res.print()
            except Exception:
                pass

            table_text = extract_table_text(res)
            if not table_text:
                # helpful debug: show what keys exist if we can
                d = _as_dict(res)
                keys = list(d.keys()) if isinstance(d, dict) else None
                print(f"[WARN] Could not extract table text for {img_path.name}. keys={keys}")
                continue

            rows = parse_table_to_rows(table_text)

            base = out_dir / img_path.stem
            write_csv(rows, base.with_suffix(".csv"))
            write_markdown(rows, base.with_suffix(".md"))

            print(f"[INFO] Saved CSV: {base.with_suffix('.csv')}")
            print(f"[INFO] Saved Markdown: {base.with_suffix('.md')}")


if __name__ == "__main__":
    main()
