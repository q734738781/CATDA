from __future__ import annotations

import argparse
import random
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union
import multiprocessing as mp

# =========================================================
# SQLite manifest (resumable)
#   - Windows friendly (no SIGALRM)
#   - Minimal schema (no last_error / last_dt)
# =========================================================

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
CREATE INDEX IF NOT EXISTS idx_status_attempts ON files(status, attempts);
"""

def open_manifest(db_path: Union[str, Path]) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # isolation_level=None => autocommit (simple + safe for CLI batch jobs)
    conn = sqlite3.connect(str(db_path), timeout=60.0, isolation_level=None)

    # WAL is fine on most local disks (Windows included), but can fail on some FS;
    # fallback to DELETE to avoid hard errors.
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.OperationalError:
        conn.execute("PRAGMA journal_mode=DELETE;")

    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA busy_timeout=60000;")
    conn.executescript(SCHEMA_SQL)
    return conn

def _file_stat(path: Path) -> Tuple[float, int]:
    st = path.stat()
    return st.st_mtime, st.st_size

def manifest_upsert_file(conn: sqlite3.Connection, path: Path) -> None:
    """Insert if missing; if file changed, reset to pending+attempts=0; repair running->pending."""
    now = time.time()
    mtime, size = _file_stat(path)
    row = conn.execute(
        "SELECT mtime,size,status FROM files WHERE path=?;",
        (str(path),),
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO files(path,mtime,size,status,attempts,updated_at) VALUES(?,?,?,?,?,?);",
            (str(path), mtime, size, STATUS_PENDING, 0, now),
        )
        return

    old_mtime, old_size, old_status = row
    changed = (old_mtime != mtime) or (old_size != size)

    if changed:
        conn.execute(
            "UPDATE files SET mtime=?,size=?,status=?,attempts=0,updated_at=? WHERE path=?;",
            (mtime, size, STATUS_PENDING, now, str(path)),
        )
    elif old_status == STATUS_RUNNING:
        conn.execute(
            "UPDATE files SET status=?,updated_at=? WHERE path=?;",
            (STATUS_PENDING, now, str(path)),
        )

def manifest_repair_running(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE files SET status=?,updated_at=? WHERE status=?;",
        (STATUS_PENDING, time.time(), STATUS_RUNNING),
    )

def manifest_select_pending(conn: sqlite3.Connection, max_attempts: int, limit: int | None = None) -> List[str]:
    q = """
    SELECT path FROM files
    WHERE (status=? OR status=?) AND attempts < ?
    ORDER BY updated_at ASC
    """
    args: list = [STATUS_PENDING, STATUS_FAILED, int(max_attempts)]
    if limit:
        q += " LIMIT ?"
        args.append(int(limit))
    return [r[0] for r in conn.execute(q, args).fetchall()]

def _mark_status(conn: sqlite3.Connection, paths: Iterable[str], status: str) -> None:
    now = time.time()
    conn.executemany(
        "UPDATE files SET status=?,updated_at=? WHERE path=?;",
        [(status, now, p) for p in paths],
    )

def _commit_results(conn: sqlite3.Connection, ok_paths: List[str], fail_paths: List[str]) -> None:
    """Commit statuses; attempts += 1 for both success/failure."""
    now = time.time()
    conn.executemany(
        "UPDATE files SET status=?,attempts=attempts+1,updated_at=? WHERE path=?;",
        [(STATUS_SUCCESS, now, p) for p in ok_paths],
    )
    conn.executemany(
        "UPDATE files SET status=?,attempts=attempts+1,updated_at=? WHERE path=?;",
        [(STATUS_FAILED, now, p) for p in fail_paths],
    )

# =========================================================
# Output paths / validation
# =========================================================

def pdf_out_dir(output_dir: Path, pdf_path: Path) -> Path:
    return output_dir / pdf_path.stem

def expected_markdown_path(pdf_path: Path, output_dir: Path) -> Path:
    return pdf_out_dir(output_dir, pdf_path) / f"{pdf_path.stem}.md"

def output_is_valid(pdf_path: Path, output_dir: Path) -> bool:
    mkd = expected_markdown_path(pdf_path, output_dir)
    try:
        if not mkd.exists() or mkd.stat().st_size < 32:
            return False
        with mkd.open("r", encoding="utf-8") as f:
            head = f.read(256)
        return head.strip() != ""
    except Exception:
        return False

def list_pdfs(in_dir: Path, recursive: bool) -> List[Path]:
    it = in_dir.rglob("*.pdf") if recursive else in_dir.glob("*.pdf")
    it2 = in_dir.rglob("*.PDF") if recursive else in_dir.glob("*.PDF")
    return sorted({*it, *it2})

# =========================================================
# PaddleOCRVL parsing
# =========================================================

_PIPELINE: Any | None = None

def _init_pipeline(vllm_url: str) -> None:
    """Child process initializer (spawn-safe)."""
    global _PIPELINE
    from paddleocr import PaddleOCRVL  # lazy import
    _PIPELINE = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url=vllm_url)

def _get_pipeline() -> Any:
    if _PIPELINE is None:
        raise RuntimeError("Pipeline not initialized in this process.")
    return _PIPELINE

def parse_pdf_to_markdown(input_file: Union[str, Path], output_dir: Union[str, Path], pipeline: Any) -> Path:
    input_path = Path(input_file)
    out_dir = Path(output_dir)

    pdf_dir = pdf_out_dir(out_dir, input_path)
    (pdf_dir / "img").mkdir(parents=True, exist_ok=True)

    output = pipeline.predict(input=str(input_path))

    markdown_pages: List[dict] = []
    path_map: dict[str, str] = {}
    images_to_save: List[Tuple[str, Any]] = []

    for res in output:
        md = res.markdown
        markdown_pages.append(md)

        images_dict = md.get("markdown_images") or {}
        for old_path, image in images_dict.items():
            new_rel = str(Path("img") / Path(old_path).name)
            path_map[old_path] = new_rel
            images_to_save.append((old_path, image))

    markdown_text = pipeline.concatenate_markdown_pages(markdown_pages)
    for old, new in path_map.items():
        markdown_text = markdown_text.replace(old, new)

    mkd_final = expected_markdown_path(input_path, out_dir)
    mkd_final.parent.mkdir(parents=True, exist_ok=True)
    mkd_final.write_text(markdown_text, encoding="utf-8")

    for old, image in images_to_save:
        dst_rel = path_map.get(old, old)
        dst = pdf_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst)
        try:
            image.close()
        except Exception:
            pass

    return mkd_final

def _worker(pdf_path: str, out_dir: str) -> Tuple[str, float, str]:
    """Worker entry. Returns (path, dt, err_msg)."""
    t0 = time.perf_counter()
    try:
        pipe = _get_pipeline()
        parse_pdf_to_markdown(Path(pdf_path), Path(out_dir), pipe)
        dt = time.perf_counter() - t0
        return pdf_path, dt, ""
    except BaseException as e:
        dt = time.perf_counter() - t0
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return pdf_path, dt, f"{type(e).__name__}: {tb}"

# =========================================================
# Batch scheduler
# =========================================================

def process_in_batches(
    conn: sqlite3.Connection,
    out_dir: Path,
    vllm_url: str,
    jobs: int,
    max_attempts: int,
    batch_size: int,
    shuffle: bool,
) -> None:
    ctx = mp.get_context("spawn")

    while True:
        pending = manifest_select_pending(conn, max_attempts=max_attempts, limit=batch_size)
        if not pending:
            break
        if shuffle:
            random.shuffle(pending)

        _mark_status(conn, pending, STATUS_RUNNING)
        print(f"Starting batch: {len(pending)} task(s), workers={jobs}")

        ok_paths: List[str] = []
        fail_paths: List[str] = []

        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=ctx,
            initializer=_init_pipeline,
            initargs=(vllm_url,),
        ) as ex:
            futures = {ex.submit(_worker, p, str(out_dir)): p for p in pending}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    path_s, dt, err = fut.result()
                except BaseException as e:
                    path_s, dt, err = p, 0.0, f"ExecutorError: {type(e).__name__}: {e}"

                pdf_path = Path(path_s)
                valid = (not err) and output_is_valid(pdf_path, out_dir)
                if valid:
                    ok_paths.append(path_s)
                    print(f"[OK]   {pdf_path}  ({dt:.2f}s)")
                else:
                    fail_paths.append(path_s)
                    msg = err or "InvalidOutput"
                    print(f"[FAIL] {pdf_path}: {msg}")

        _commit_results(conn, ok_paths, fail_paths)
        left = len(manifest_select_pending(conn, max_attempts=max_attempts))
        print(f"Batch done. ok={len(ok_paths)} fail={len(fail_paths)} pending={left}")

    remain = len(manifest_select_pending(conn, max_attempts=max_attempts))
    if remain:
        print(f"All batches complete, but {remain} file(s) still pending after reaching max attempts.")

# =========================================================
# CLI
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF-to-Markdown with PaddleOCRVL. SQLite manifest, resumable."
    )
    parser.add_argument("input", help="Path to a PDF file or a directory of PDFs.")
    parser.add_argument("-o", "--output", default="output", help="Output directory.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("-v", "--vllm-url", default="http://127.0.0.1:8118/v1", help="URL of the VLLM server.")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Parallel workers (processes).")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per file before giving up.")
    parser.add_argument("--batch-size", type=int, default=500, help="Tasks per batch before pool recycle.")
    parser.add_argument("--manifest", default=None, help="Path to manifest SQLite DB, default <output>/manifest.db")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle order inside each batch.")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip files whose outputs already look valid.")
    args = parser.parse_args()

    # On Windows this is the default; on Linux it avoids fork issues with native libs.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else (output_dir / "manifest.db")
    conn = open_manifest(manifest_path)
    manifest_repair_running(conn)

    if input_path.is_file():
        from paddleocr import PaddleOCRVL
        pipe = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url=args.vllm_url)

        manifest_upsert_file(conn, input_path)

        if args.skip_if_exists and output_is_valid(input_path, output_dir):
            _mark_status(conn, [str(input_path)], STATUS_SUCCESS)
            print("Output already exists and looks valid. Skipping.")
            return

        _mark_status(conn, [str(input_path)], STATUS_RUNNING)
        t0 = time.perf_counter()
        try:
            parse_pdf_to_markdown(input_path, output_dir, pipe)
            dt = time.perf_counter() - t0
            if output_is_valid(input_path, output_dir):
                _commit_results(conn, [str(input_path)], [])
                print(f"[OK]   {input_path}  ({dt:.2f}s)")
                return
            _commit_results(conn, [], [str(input_path)])
            print("[FAIL] InvalidOutput", file=sys.stderr)
            sys.exit(2)
        except BaseException as e:
            dt = time.perf_counter() - t0
            _commit_results(conn, [], [str(input_path)])
            print(f"[FAIL] {type(e).__name__}: {e}  ({dt:.2f}s)", file=sys.stderr)
            sys.exit(1)

    if input_path.is_dir():
        pdfs = list_pdfs(input_path, args.recursive)
        if not pdfs:
            print("No PDFs found to process.")
            return

        # sync manifest with filesystem
        for p in pdfs:
            manifest_upsert_file(conn, p)

        if args.skip_if_exists:
            ok_paths = [str(p) for p in pdfs if output_is_valid(p, output_dir)]
            if ok_paths:
                _mark_status(conn, ok_paths, STATUS_SUCCESS)

        jobs = max(1, int(args.jobs))
        if jobs == 1:
            from paddleocr import PaddleOCRVL
            pipe = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url=args.vllm_url)

            for p_str in manifest_select_pending(conn, max_attempts=args.max_attempts):
                p = Path(p_str)
                _mark_status(conn, [p_str], STATUS_RUNNING)
                t0 = time.perf_counter()
                try:
                    parse_pdf_to_markdown(p, output_dir, pipe)
                    dt = time.perf_counter() - t0
                    if output_is_valid(p, output_dir):
                        _commit_results(conn, [p_str], [])
                        print(f"[OK]   {p}  ({dt:.2f}s)")
                    else:
                        _commit_results(conn, [], [p_str])
                        print(f"[FAIL] {p}: InvalidOutput")
                except BaseException as e:
                    dt = time.perf_counter() - t0
                    _commit_results(conn, [], [p_str])
                    print(f"[FAIL] {p}: {type(e).__name__}: {e}  ({dt:.2f}s)")
        else:
            process_in_batches(
                conn=conn,
                out_dir=output_dir,
                vllm_url=args.vllm_url,
                jobs=jobs,
                max_attempts=args.max_attempts,
                batch_size=max(1, args.batch_size),
                shuffle=args.shuffle,
            )
        return

    raise FileNotFoundError(f"Input path not found: {input_path}")

if __name__ == "__main__":
    started = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    finally:
        print(f"Total Time taken: {time.time() - started:.2f} seconds")
