#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF -> Markdown (single .md per PDF) using PaddleOCR-VL (PaddleOCRVL).

Design goals (practical + production-ish):
- Keep "client (layout + orchestration)" and "VLM server (vLLM)" separated via HTTP.
- Archive all images referenced in the generated Markdown.
- Optional chart parsing (use_chart_recognition) via a CLI switch.
- Resumable batch processing via a minimal SQLite manifest (no last_error / last_dt).
- Windows-friendly: no signal alarms; no multiprocessing required.

Throughput note (important):
- The VLM server can be underutilized if you process PDFs strictly one-by-one.
- PaddleOCR docs state the client will group sub-images from *single or multiple* inputs and send
  concurrent requests to the VLM server; thus we support batching multiple PDFs per `predict()` call
  to smooth utilization without spawning multiple pipeline instances.

Docs assumptions (PaddleOCR-VL):
- Each page result prints/contains `input_path` and (for PDF) `page_index`.
- Each page result has `.markdown` as a dict with keys: markdown_texts / markdown_images / page_continuation_flags.

If your PaddleOCR version differs, this script includes fallbacks and will automatically fall back
to per-PDF processing if batch mode isn't supported.

References:
- Performance tuning: client concurrency and grouping sub-images from single or multiple inputs.
- Service deployment API: PDF 10-page default limit and how to remove it.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
# -----------------------------
# Manifest (SQLite)
# -----------------------------
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  mtime REAL NOT NULL,
  size  INTEGER NOT NULL,
  status TEXT NOT NULL,               -- pending | running | success | failed
  attempts INTEGER NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_status_attempts ON files(status, attempts);
"""


def _now() -> float:
    return time.time()


def open_manifest(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0, isolation_level=None)

    # WAL usually improves throughput; fallback for FS that don't support it well (some Windows setups)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        conn.execute("PRAGMA journal_mode=DELETE;")

    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA busy_timeout=60000;")
    conn.executescript(SCHEMA_SQL)
    return conn


def _file_stat(p: Path) -> Tuple[float, int]:
    st = p.stat()
    return float(st.st_mtime), int(st.st_size)


def manifest_repair_running(conn: sqlite3.Connection) -> None:
    """If the script crashed mid-run, move running -> pending."""
    conn.execute(
        "UPDATE files SET status=?, updated_at=? WHERE status=?;",
        (STATUS_PENDING, _now(), STATUS_RUNNING),
    )


def manifest_upsert_file(conn: sqlite3.Connection, pdf_path: Path) -> None:
    """
    Insert missing record.
    If file changed: reset (pending + attempts=0).
    If status=running (leftover): reset to pending.
    """
    mtime, size = _file_stat(pdf_path)
    now = _now()

    row = conn.execute(
        "SELECT mtime, size, status FROM files WHERE path=?;",
        (str(pdf_path),),
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO files(path, mtime, size, status, attempts, updated_at) VALUES(?,?,?,?,?,?);",
            (str(pdf_path), mtime, size, STATUS_PENDING, 0, now),
        )
        return

    old_mtime, old_size, old_status = float(row[0]), int(row[1]), str(row[2])
    changed = (old_mtime != mtime) or (old_size != size)

    if changed:
        conn.execute(
            "UPDATE files SET mtime=?, size=?, status=?, attempts=0, updated_at=? WHERE path=?;",
            (mtime, size, STATUS_PENDING, now, str(pdf_path)),
        )
    elif old_status == STATUS_RUNNING:
        conn.execute(
            "UPDATE files SET status=?, updated_at=? WHERE path=?;",
            (STATUS_PENDING, now, str(pdf_path)),
        )


def manifest_select_pending(conn: sqlite3.Connection, max_attempts: int, limit: Optional[int] = None) -> List[str]:
    q = """
    SELECT path FROM files
    WHERE (status=? OR status=?) AND attempts < ?
    ORDER BY updated_at ASC
    """
    args: List[Any] = [STATUS_PENDING, STATUS_FAILED, int(max_attempts)]
    if limit is not None:
        q += " LIMIT ?"
        args.append(int(limit))
    rows = conn.execute(q, args).fetchall()
    return [str(r[0]) for r in rows]


def _mark_status(conn: sqlite3.Connection, path: str, status: str) -> None:
    conn.execute(
        "UPDATE files SET status=?, updated_at=? WHERE path=?;",
        (status, _now(), path),
    )


def _commit_ok(conn: sqlite3.Connection, path: str) -> None:
    conn.execute(
        "UPDATE files SET status=?, attempts=attempts+1, updated_at=? WHERE path=?;",
        (STATUS_SUCCESS, _now(), path),
    )


def _commit_fail(conn: sqlite3.Connection, path: str) -> None:
    conn.execute(
        "UPDATE files SET status=?, attempts=attempts+1, updated_at=? WHERE path=?;",
        (STATUS_FAILED, _now(), path),
    )


# -----------------------------
# I/O conventions
# -----------------------------
@dataclass(frozen=True)
class OutputLayout:
    base_dir: Path          # output/<rel_parent>/<stem>/
    md_path: Path           # output/<rel_parent>/<stem>/<stem>.md
    img_dir: Path           # output/<rel_parent>/<stem>/img/


def compute_output_layout(pdf_path: Path, input_root: Path, output_root: Path) -> OutputLayout:
    """
    Keep outputs stable & collision-free by mirroring the input directory structure:

      input_root/
        a/b/c.pdf

      output_root/
        a/b/c/
          c.md
          img/...

    For single-file input, input_root defaults to pdf_path.parent.
    """
    rel = pdf_path.relative_to(input_root)
    base_dir = (output_root / rel.parent / rel.stem).resolve()
    img_dir = base_dir / "img"
    md_path = base_dir / f"{rel.stem}.md"
    return OutputLayout(base_dir=base_dir, md_path=md_path, img_dir=img_dir)


def output_is_valid(layout: OutputLayout) -> bool:
    try:
        if not layout.md_path.exists() or layout.md_path.stat().st_size < 32:
            return False
        head = layout.md_path.read_text(encoding="utf-8")[:256].strip()
        return head != ""
    except Exception:
        return False


def list_pdfs(in_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        pdfs = list(in_dir.rglob("*.pdf")) + list(in_dir.rglob("*.PDF"))
    else:
        pdfs = list(in_dir.glob("*.pdf")) + list(in_dir.glob("*.PDF"))
    # Deduplicate and sort
    uniq = sorted({p.resolve() for p in pdfs})
    return uniq


# -----------------------------
# Result helpers (robust across versions)
# -----------------------------
def _get_result_input_path(res: Any) -> Optional[str]:
    """
    Extract input_path from a PaddleOCR result object/dict.
    PaddleOCR-VL docs describe `input_path` in printed output.
    """
    if hasattr(res, "input_path"):
        v = getattr(res, "input_path")
        if isinstance(v, str) and v:
            return v
    if isinstance(res, dict):
        v = res.get("input_path")
        if isinstance(v, str) and v:
            return v
    # Try JSON-ish access patterns
    for attr in ("json", "to_json"):
        obj = getattr(res, attr, None)
        try:
            if callable(obj):
                j = obj()
            else:
                j = obj
        except Exception:
            continue
        if isinstance(j, dict):
            v = j.get("input_path") or j.get("inputPath")
            if isinstance(v, str) and v:
                return v
    return None


def _get_result_markdown(res: Any) -> Dict[str, Any]:
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        return md
    if isinstance(res, dict) and isinstance(res.get("markdown"), dict):
        return res["markdown"]
    raise TypeError(f"Unexpected result.markdown type: {type(md)}")


_TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table>", flags=re.IGNORECASE | re.DOTALL)

def _clean_one_table_html(table_html: str) -> str:
    """
    Clean one <table>...</table> HTML fragment:
    - Keep table structure tags
    - Keep only rowspan/colspan attrs on td/th
    - Remove style/class/border/width/... noise
    """
    # Fast fallback if bs4 isn't installed
    try:
        from bs4 import BeautifulSoup
    except Exception:
        t = table_html
        # Remove common noisy attributes (keep rowspan/colspan by NOT touching them)
        t = re.sub(r"\sstyle=(\"[^\"]*\"|'[^']*')", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\sclass=(\"[^\"]*\"|'[^']*')", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\sborder=(\"[^\"]*\"|'[^']*'|[0-9]+)", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\swidth=(\"[^\"]*\"|'[^']*'|[0-9%]+)", "", t, flags=re.IGNORECASE)
        return t

    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return table_html

    allowed_tags = {
        "table", "thead", "tbody", "tfoot",
        "tr", "th", "td", "caption",
        # inline tags that sometimes appear in cells
        "br", "sub", "sup",
    }
    allowed_attrs = {
        "td": {"rowspan", "colspan"},
        "th": {"rowspan", "colspan"},
    }

    # Clean subtree
    for tag in list(table.find_all(True)):
        name = (tag.name or "").lower()

        # Drop/unwrap unexpected tags but keep their textual content
        if name not in allowed_tags:
            tag.unwrap()
            continue

        keep = allowed_attrs.get(name, set())
        if keep:
            tag.attrs = {k: v for k, v in tag.attrs.items() if k in keep}
        else:
            tag.attrs = {}  # remove all attrs for non td/th tags

    # Ensure <table> has no attrs
    table.attrs = {}

    return str(table)


def clean_html_tables_in_markdown(md_text: str) -> str:
    """Replace each HTML <table> block with a cleaned version (keep merged cell semantics)."""
    def _repl(m: re.Match) -> str:
        return _clean_one_table_html(m.group(0))
    return _TABLE_BLOCK_RE.sub(_repl, md_text)


def postprocess_markdown_file(md_path: Path) -> None:
    """Read -> clean HTML tables -> write back (only if changed)."""
    try:
        raw = md_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return

    cleaned = clean_html_tables_in_markdown(raw)
    if cleaned != raw:
        md_path.write_text(cleaned, encoding="utf-8")

# -----------------------------
# Markdown materialization (streaming)
# -----------------------------
def _ensure_ext_png(name: str) -> str:
    p = Path(name)
    if p.suffix:
        return name
    return name + ".png"


@dataclass
class PdfAccumulator:
    pdf_path: Path
    layout: OutputLayout
    markdown_pages: List[Dict[str, Any]] = field(default_factory=list)
    path_map: Dict[str, str] = field(default_factory=dict)          # old_norm -> new_rel
    used_names: Dict[str, int] = field(default_factory=dict)        # filename -> count

    def add_page(self, md: Dict[str, Any]) -> None:
        """
        Consume one page markdown dict:
        - Save images immediately and record path rewrites
        - Store a *lightweight* page md dict (without PIL images) for concatenation
        """
        self.layout.base_dir.mkdir(parents=True, exist_ok=True)
        self.layout.img_dir.mkdir(parents=True, exist_ok=True)

        images_dict = md.get("markdown_images") or {}
        if isinstance(images_dict, dict):
            for old_path, image in images_dict.items():
                if not isinstance(old_path, str):
                    continue
                old_norm = old_path.replace("\\", "/")
                if old_norm in self.path_map:
                    continue

                base_name = Path(old_norm).name
                base_name = _ensure_ext_png(base_name)

                # Deduplicate names if needed
                count = self.used_names.get(base_name, 0)
                if count == 0:
                    final_name = base_name
                else:
                    stem = Path(base_name).stem
                    suf = Path(base_name).suffix
                    final_name = f"{stem}_{count}{suf}"
                self.used_names[base_name] = count + 1

                new_rel = str(Path("img") / final_name).replace("\\", "/")
                self.path_map[old_norm] = new_rel

                dst = self.layout.base_dir / new_rel
                dst.parent.mkdir(parents=True, exist_ok=True)

                # Save and close asap to bound memory/FD usage
                try:
                    image.save(dst)
                finally:
                    close = getattr(image, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass

        # Store lightweight md for concatenation
        md_light = dict(md)
        if "markdown_images" in md_light:
            md_light["markdown_images"] = {}
        self.markdown_pages.append(md_light)

    def finalize(self, pipeline: Any) -> None:
        concat = getattr(pipeline, "concatenate_markdown_pages", None)
        markdown_text = ""
        if callable(concat):
            try:
                markdown_text = concat(self.markdown_pages)
            except Exception:
                markdown_text = ""

        if not markdown_text:
            # Fallback: concatenate markdown_texts ourselves
            parts: List[str] = []
            for md in self.markdown_pages:
                txt = md.get("markdown_texts")
                if isinstance(txt, list):
                    parts.append("\n".join([str(x) for x in txt]))
                elif isinstance(txt, str):
                    parts.append(txt)
                else:
                    parts.append("")
            markdown_text = "\n\n".join(parts)

        # Rewrite image paths to our output layout
        for old_norm, new_rel in self.path_map.items():
            markdown_text = markdown_text.replace(old_norm, new_rel)
            markdown_text = markdown_text.replace(old_norm.replace("/", "\\"), new_rel)

        self.layout.md_path.parent.mkdir(parents=True, exist_ok=True)
        self.layout.md_path.write_text(markdown_text, encoding="utf-8")


# -----------------------------
# Pipeline building and predict wrappers
# -----------------------------
def build_pipeline(
    *,
    vllm_url: str,
    vl_rec_api_model_name: Optional[str],
    vl_rec_api_key: Optional[str],
    vl_rec_max_concurrency: Optional[int],
    device: Optional[str],
    precision: Optional[str],
) -> Any:
    """
    Create PaddleOCRVL pipeline once.

    Note:
    Even with a remote VLM server, the client still runs local sequential models for layout detection.
    Avoid creating many pipeline instances unless you intentionally want multi-instance parallelism.
    """
    from paddleocr import PaddleOCRVL  # lazy import

    kwargs: Dict[str, Any] = {
        "vl_rec_backend": "vllm-server",
        "vl_rec_server_url": vllm_url,
    }
    if vl_rec_api_model_name:
        kwargs["vl_rec_api_model_name"] = vl_rec_api_model_name
    if vl_rec_api_key:
        kwargs["vl_rec_api_key"] = vl_rec_api_key
    if vl_rec_max_concurrency is not None:
        kwargs["vl_rec_max_concurrency"] = int(vl_rec_max_concurrency)
    if device:
        kwargs["device"] = device
    if precision:
        kwargs["precision"] = precision

    return PaddleOCRVL(**kwargs)


def _predict_with_fallback(pipeline: Any, inputs: List[str], kwargs: Dict[str, Any]) -> Iterable[Any]:
    """
    Try a few calling conventions because different PaddleOCR versions may accept:
      - predict(input=[...], ...)
      - predict([...], ...)
    """
    # 1) keyword input=
    try:
        return pipeline.predict(input=inputs, **kwargs)
    except TypeError:
        pass
    # 2) positional
    return pipeline.predict(inputs, **kwargs)


# -----------------------------
# Batch processing logic
# -----------------------------
def process_pdf_batch(
    *,
    pipeline: Any,
    pdf_paths: List[Path],
    input_root: Path,
    output_root: Path,
    use_chart_recognition: bool,
    use_queues: bool,
) -> Dict[str, Tuple[bool, str, Optional[OutputLayout]]]:
    """
    Process a batch of PDFs in ONE `pipeline.predict()` call (preferred for throughput),
    then split results back into per-PDF outputs.

    Returns:
      dict[pdf_str] = (ok, message, layout)
    """
    # Prepare accumulators
    accs: Dict[str, PdfAccumulator] = {}
    key_to_pdf: Dict[str, Path] = {}

    for p in pdf_paths:
        layout = compute_output_layout(p, input_root=input_root, output_root=output_root)
        key = str(p.resolve())
        accs[key] = PdfAccumulator(pdf_path=p, layout=layout)
        key_to_pdf[key] = p

    inputs = [str(p) for p in pdf_paths]
    kwargs: Dict[str, Any] = {
        "use_chart_recognition": bool(use_chart_recognition),
        "use_queues": bool(use_queues),
    }

    # Run prediction (stream results)
    results_iter = _predict_with_fallback(pipeline, inputs, kwargs)

    seen: Dict[str, int] = {k: 0 for k in accs.keys()}
    for res in results_iter:
        ip = _get_result_input_path(res)
        if not ip:
            # Can't route; ignore and continue. We'll fail missing ones later.
            continue
        k = str(Path(ip).resolve())
        if k not in accs:
            # Some versions might return the input path exactly as passed (maybe relative).
            # Try a softer match.
            k2 = str(Path(ip).absolute())
            if k2 in accs:
                k = k2
            else:
                continue

        md = _get_result_markdown(res)
        accs[k].add_page(md)
        seen[k] = seen.get(k, 0) + 1

    # Finalize per PDF
    out: Dict[str, Tuple[bool, str, Optional[OutputLayout]]] = {}
    for k, acc in accs.items():
        if seen.get(k, 0) <= 0:
            out[str(acc.pdf_path)] = (False, "NoResults", acc.layout)
            continue
        try:
            acc.finalize(pipeline)
            if output_is_valid(acc.layout):
                out[str(acc.pdf_path)] = (True, "OK", acc.layout)
            else:
                out[str(acc.pdf_path)] = (False, "InvalidOutput", acc.layout)
        except Exception as e:
            msg = "".join(traceback.format_exception_only(type(e), e)).strip()
            out[str(acc.pdf_path)] = (False, msg, acc.layout)

    return out


def process_one_pdf_sequential(
    *,
    pipeline: Any,
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    use_chart_recognition: bool,
    use_queues: bool,
) -> Tuple[bool, str, Optional[OutputLayout]]:
    layout = compute_output_layout(pdf_path, input_root=input_root, output_root=output_root)
    kwargs: Dict[str, Any] = {
        "input": str(pdf_path),
        "use_chart_recognition": bool(use_chart_recognition),
        "use_queues": bool(use_queues),
    }
    try:
        results = pipeline.predict(**kwargs)
        acc = PdfAccumulator(pdf_path=pdf_path, layout=layout)
        for res in results:
            md = _get_result_markdown(res)
            acc.add_page(md)
        acc.finalize(pipeline)
        if output_is_valid(layout):
            return True, "OK", layout
        return False, "InvalidOutput", layout
    except Exception as e:
        msg = "".join(traceback.format_exception_only(type(e), e)).strip()
        return False, msg, layout


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="PDF->Markdown via PaddleOCR-VL (client uses vLLM server).")
    ap.add_argument("input", help="PDF file or directory containing PDFs.")
    ap.add_argument("-o", "--output", default="output", help="Output directory root.")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories (for directory input).")

    ap.add_argument("--manifest", default=None, help="SQLite manifest path (default: <output>/manifest.db).")
    ap.add_argument("--max-attempts", type=int, default=3, help="Max attempts per PDF before giving up.")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N pending PDFs (debug).")
    ap.add_argument("--skip-if-exists", action="store_true", help="Skip PDFs whose output md already looks valid.")

    # Client <-> VLM server wiring
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8118/v1", help="vLLM OpenAI-compatible base URL.")
    ap.add_argument("--vl-rec-api-model-name", default=None, help="Model name for OpenAI-compatible API (optional).")
    ap.add_argument("--vl-rec-api-key", default=None, help="API key for OpenAI-compatible API (optional).")

    # Performance knobs (client-side)
    ap.add_argument("--vl-rec-max-concurrency", type=int, default=512,
                    help="Max concurrent requests sent to VLM server (client-side). Tune with server max-num-seqs.")
    ap.add_argument("--pdf-batch-size", type=int, default=4,
                    help="How many PDFs to pack into ONE pipeline.predict() call (>=1). "
                         "Higher smooths GPU utilization but increases memory pressure.")
    ap.add_argument("--no-queues", action="store_true",
                    help="Disable internal async queues (usually keep enabled for multi-page PDFs).")

    # Feature knobs
    ap.add_argument("--expand-charts", action="store_true",
                    help="Enable chart parsing (use_chart_recognition=True). Off by default.")
    ap.add_argument("--clean-html-tables", action="store_true",
                help="Post-process .md: strip noisy attrs from HTML <table>, keep rowspan/colspan.")


    # Local (client) inference config
    ap.add_argument("--device", default=None,
                    help="Client device for local modules (e.g. gpu:0, cpu).")
    ap.add_argument("--precision", default=None,
                    help="Client precision for local modules (e.g. fp16, fp32).")

    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else (out_root / "manifest.db")
    conn = open_manifest(manifest_path)
    manifest_repair_running(conn)

    # Define input_root for stable output layout mapping
    if in_path.is_file():
        input_root = in_path.parent
        pdfs = [in_path]
    elif in_path.is_dir():
        input_root = in_path
        pdfs = list_pdfs(in_path, recursive=bool(args.recursive))
        if not pdfs:
            print("No PDFs found.", file=sys.stderr)
            return
    else:
        raise FileNotFoundError(f"Input path not found: {in_path}")

    # Sync manifest
    for p in pdfs:
        manifest_upsert_file(conn, p)

    # Optional fast skip
    if args.skip_if_exists:
        for p in pdfs:
            layout = compute_output_layout(p, input_root=input_root, output_root=out_root)
            if output_is_valid(layout):
                _mark_status(conn, str(p), STATUS_SUCCESS)

    # Create pipeline ONCE (important to avoid GPU memory blow-up)
    pipeline = build_pipeline(
        vllm_url=args.vllm_url,
        vl_rec_api_model_name=args.vl_rec_api_model_name,
        vl_rec_api_key=args.vl_rec_api_key,
        vl_rec_max_concurrency=(int(args.vl_rec_max_concurrency) if args.vl_rec_max_concurrency else None),
        device=args.device,
        precision=args.precision,
    )

    pending = manifest_select_pending(conn, max_attempts=int(args.max_attempts), limit=args.limit)
    if not pending:
        print("Nothing to do.")
        return

    use_queues = not bool(args.no_queues)
    batch_size = max(1, int(args.pdf_batch_size))

    # Process in batches for smoother server utilization
    i = 0
    while i < len(pending):
        batch_paths = [Path(p) for p in pending[i:i + batch_size]]
        i += batch_size

        # Mark running up-front
        for p in batch_paths:
            _mark_status(conn, str(p), STATUS_RUNNING)

        t0 = time.perf_counter()
        try:
            # Try batch mode first (better throughput)
            batch_out = process_pdf_batch(
                pipeline=pipeline,
                pdf_paths=batch_paths,
                input_root=input_root,
                output_root=out_root,
                use_chart_recognition=bool(args.expand_charts),
                use_queues=use_queues,
            )
        except TypeError as e:
            # Likely "predict doesn't accept list input" in this version.
            # Fall back to per-PDF sequential processing.
            batch_out = {}
            for p in batch_paths:
                ok, msg, layout = process_one_pdf_sequential(
                    pipeline=pipeline,
                    pdf_path=p,
                    input_root=input_root,
                    output_root=out_root,
                    use_chart_recognition=bool(args.expand_charts),
                    use_queues=use_queues,
                )
                batch_out[str(p)] = (ok, msg, layout)
        except Exception:
            # For any other batch-level failure, try sequential to salvage per-file progress.
            batch_out = {}
            for p in batch_paths:
                ok, msg, layout = process_one_pdf_sequential(
                    pipeline=pipeline,
                    pdf_path=p,
                    input_root=input_root,
                    output_root=out_root,
                    use_chart_recognition=bool(args.expand_charts),
                    use_queues=use_queues,
                )
                batch_out[str(p)] = (ok, msg, layout)

        dt = time.perf_counter() - t0

        # Commit per-file
        for p in batch_paths:
            key = str(p)
            ok, msg, layout = batch_out.get(key, (False, "BatchMissingResult", None))
            if ok:
                _commit_ok(conn, key)

                # ---- NEW: clean HTML tables in the produced markdown (optional) ----
                if getattr(args, "clean_html_tables", False) and layout is not None:
                    try:
                        postprocess_markdown_file(layout.md_path)
                    except Exception as e:
                        # Don't fail the whole pdf just because postprocess failed
                        print(f"[WARN] clean-html failed: {layout.md_path}: {e}", file=sys.stderr)
                # -------------------------------------------------------------------

                outp = layout.md_path if layout else "(unknown)"
                print(f"[OK]   {p}  ->  {outp}")

            else:
                _commit_fail(conn, key)
                print(f"[FAIL] {p}: {msg}", file=sys.stderr)

        print(f"[BATCH] {len(batch_paths)} file(s) done in {dt:.2f}s")

    remain = manifest_select_pending(conn, max_attempts=int(args.max_attempts))
    if remain:
        print(f"Done. Still pending (retries left): {len(remain)} file(s).", file=sys.stderr)
    else:
        print("Done. All PDFs processed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

"""
Example:
python paddleocr_vl_pdf2md_client.py ../data/Papers -o ../data/Papers_output \
  --vllm-url http://127.0.0.1:8118/v1 \
  --vl-rec-max-concurrency 128 \
  --pdf-batch-size 4 \
  --expand-charts
"""
