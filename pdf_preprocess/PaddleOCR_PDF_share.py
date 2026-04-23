from __future__ import annotations

import argparse
import copy
import html
import os
import re
import sys
import time
import signal
import sqlite3
import traceback
import random
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# =======================
# SQLite 清单
# =======================

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  mtime REAL NOT NULL,
  size  INTEGER NOT NULL,
  status TEXT NOT NULL,               -- pending | running | success | failed
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  last_dt REAL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status ON files(status);
"""

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED  = "failed"

def open_manifest(db_path: Union[str, Path]) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60.0, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-262144;")   # 约 1GB 页缓存上限，按需调整
    conn.execute("PRAGMA wal_autocheckpoint=1000;")
    conn.execute("PRAGMA busy_timeout=60000;")
    conn.executescript(SCHEMA_SQL)
    return conn

def file_stat(path: Path) -> Tuple[float, int]:
    st = path.stat()
    return st.st_mtime, st.st_size

def manifest_upsert_file(conn: sqlite3.Connection, path: Path) -> None:
    now = time.time()
    mtime, size = file_stat(path)
    row = conn.execute("SELECT mtime,size,status FROM files WHERE path=?;", (str(path),)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO files(path,mtime,size,status,attempts,updated_at) VALUES(?,?,?,?,?,?);",
            (str(path), mtime, size, STATUS_PENDING, 0, now),
        )
    else:
        old_mtime, old_size, old_status = row
        changed = (old_mtime != mtime) or (old_size != size)
        if changed:
            conn.execute(
                "UPDATE files SET mtime=?,size=?,status=?,attempts=0,last_error=NULL,last_dt=NULL,updated_at=? WHERE path=?;",
                (mtime, size, STATUS_PENDING, now, str(path)),
            )
        elif old_status == STATUS_RUNNING:
            conn.execute("UPDATE files SET status=?,updated_at=? WHERE path=?;", (STATUS_PENDING, now, str(path)))

def manifest_bulk_mark_running(conn: sqlite3.Connection, paths: List[str]) -> None:
    now = time.time()
    with conn:
        for p in paths:
            conn.execute("UPDATE files SET status=?,updated_at=? WHERE path=?;", (STATUS_RUNNING, now, p))

def manifest_bulk_commit_results(
    conn: sqlite3.Connection,
    ok_items: List[Tuple[str, float]],
    fail_items: List[Tuple[str, float, str]],
) -> None:
    now = time.time()
    with conn:
        for p, dt in ok_items:
            conn.execute(
                "UPDATE files SET status=?,attempts=attempts+1,last_dt=?,last_error=NULL,updated_at=? WHERE path=?;",
                (STATUS_SUCCESS, float(dt), now, p),
            )
        for p, dt, err in fail_items:
            conn.execute(
                "UPDATE files SET status=?,attempts=attempts+1,last_dt=?,last_error=?,updated_at=? WHERE path=?;",
                (STATUS_FAILED, float(dt), err, now, p),
            )

def manifest_select_pending(conn: sqlite3.Connection, max_attempts: int, limit: int | None = None) -> List[str]:
    q = """
    SELECT path FROM files
    WHERE (status=? OR status=?) AND attempts < ?
    ORDER BY updated_at ASC
    """
    args = [STATUS_PENDING, STATUS_FAILED, max_attempts]
    if limit:
        q += " LIMIT ?"
        args.append(limit)
    rows = conn.execute(q, args).fetchall()
    return [r[0] for r in rows]

def manifest_repair_running(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE files SET status=?,updated_at=? WHERE status=?;", (STATUS_PENDING, time.time(), STATUS_RUNNING))

# =======================
# 输出与有效性
# =======================

def pdf_out_dir(output_dir: Path, pdf_path: Path) -> Path:
    return Path(output_dir) / pdf_path.stem

def expected_markdown_path(pdf_path: Path, output_dir: Path) -> Path:
    return pdf_out_dir(output_dir, pdf_path) / f"{pdf_path.stem}.md"

def output_is_valid(pdf_path: Path, output_dir: Path) -> bool:
    mkd = expected_markdown_path(pdf_path, output_dir)
    try:
        if not mkd.exists() or mkd.stat().st_size < 32:
            return False
        with open(mkd, "r", encoding="utf-8") as f:
            head = f.read(64)
            if head.strip() == "":
                return False
        return True
    except Exception:
        return False

def list_pdfs(in_dir: Path, recursive: bool) -> List[Path]:
    pats = ["*.pdf", "*.PDF"]
    pdfs: List[Path] = []
    for pat in pats:
        pdfs.extend(in_dir.rglob(pat) if recursive else in_dir.glob(pat))
    return sorted(set(pdfs))

# =======================
# PaddleOCRVL 解析
# =======================

_PIPELINE = None  # 子进程内
_CHILD_TIMEOUT_S = 0
_TABLE_FORMAT = "raw"
_CHART_OUTPUT = "off"
_TABLE_BLOCK_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)

def normalize_vllm_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url:
        raise ValueError("VLM server URL cannot be empty.")
    if "://" not in url:
        url = f"http://{url}"
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url

def check_remote_server(vllm_url: str, timeout_s: float = 3.0) -> None:
    base_url = vllm_url[:-3] if vllm_url.endswith("/v1") else vllm_url
    health_url = f"{base_url}/health"
    models_url = f"{vllm_url}/models"

    try:
        with urlopen(health_url, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                raise RuntimeError(f"Health endpoint returned HTTP {status}: {health_url}")
    except HTTPError as e:
        raise RuntimeError(f"Remote server health check failed: HTTP {e.code} at {health_url}") from e
    except URLError as e:
        raise RuntimeError(
            f"Cannot reach remote VLM server at {health_url}. "
            "You can run this script locally against a remote server, but the server must be reachable."
        ) from e

    try:
        with urlopen(models_url, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                raise RuntimeError(f"Models endpoint returned HTTP {status}: {models_url}")
    except HTTPError:
        # Some deployments expose /health but not /v1/models; don't block on that.
        pass
    except URLError:
        pass

def ensure_no_proxy_for_vllm_url(vllm_url: str) -> None:
    host = urlparse(vllm_url).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "").strip()
        items = [item.strip() for item in current.split(",") if item.strip()]
        if host not in items:
            items.append(host)
            os.environ[key] = ",".join(items)

def import_paddleocr_vl():
    try:
        from paddleocr import PaddleOCRVL
        return PaddleOCRVL
    except Exception as e:
        raise RuntimeError(
            "Failed to import local PaddleOCR client. "
            "You do not need to upload this script to the remote inference machine, "
            "but you do need a working local PaddleOCR client environment to call the remote VLM server."
        ) from e

class _HTMLTableCleaner(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        attrs_out: List[Tuple[str, str]] = []
        tag_lower = tag.lower()
        if tag_lower in {"td", "th"}:
            for key, value in attrs:
                if key in {"rowspan", "colspan"} and value is not None:
                    attrs_out.append((key, value))
        attr_str = "".join(
            f' {key}="{html.escape(value, quote=True)}"' for key, value in attrs_out
        )
        self.parts.append(f"<{tag_lower}{attr_str}>")

    def handle_startendtag(
        self, tag: str, attrs: List[Tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in {"br"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(f"</{tag.lower()}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        return None

    def get_html(self) -> str:
        return "".join(self.parts)

class _HTMLTableToMarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[str]] = []
        self._current_row: List[str] | None = None
        self._cell_parts: List[str] | None = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "tr":
            self._current_row = []
        elif tag_lower in {"td", "th"}:
            self._cell_parts = []
        elif tag_lower == "br" and self._cell_parts is not None:
            self._cell_parts.append("<br>")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"td", "th"} and self._cell_parts is not None:
            text = "".join(self._cell_parts)
            text = re.sub(r"\s+", " ", text).strip()
            assert self._current_row is not None
            self._current_row.append(text)
            self._cell_parts = None
        elif tag_lower == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

def clean_html_tables(markdown_text: str) -> str:
    def _clean(match: re.Match[str]) -> str:
        parser = _HTMLTableCleaner()
        parser.feed(match.group(0))
        parser.close()
        return parser.get_html()

    return _TABLE_BLOCK_RE.sub(_clean, markdown_text)

def _escape_md_cell(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", "<br>").strip()

def html_tables_to_markdown(markdown_text: str) -> str:
    def _to_md(match: re.Match[str]) -> str:
        parser = _HTMLTableToMarkdownParser()
        parser.feed(match.group(0))
        parser.close()
        rows = parser.rows
        if not rows:
            return match.group(0)
        col_count = max(len(row) for row in rows)
        norm_rows = [row + [""] * (col_count - len(row)) for row in rows]
        header = [_escape_md_cell(cell) for cell in norm_rows[0]]
        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * col_count) + " |",
        ]
        for row in norm_rows[1:]:
            md_lines.append("| " + " | ".join(_escape_md_cell(cell) for cell in row) + " |")
        return "\n" + "\n".join(md_lines) + "\n"

    return _TABLE_BLOCK_RE.sub(_to_md, markdown_text)

def transform_markdown_text(markdown_text: str, table_format: str) -> str:
    if table_format == "raw":
        return markdown_text
    cleaned = clean_html_tables(markdown_text)
    if table_format == "clean":
        return cleaned
    if table_format == "markdown":
        return html_tables_to_markdown(cleaned)
    raise ValueError(f"Unsupported table format: {table_format}")

def render_markdown_info(
    res: Any,
    table_format: str,
    chart_as_table: bool,
) -> Dict[str, Any]:
    original_flag = res["model_settings"].get("use_chart_recognition", False)
    res["model_settings"]["use_chart_recognition"] = chart_as_table
    try:
        pretty = (table_format == "raw")
        md_info = res._to_markdown(pretty=pretty)
    finally:
        res["model_settings"]["use_chart_recognition"] = original_flag
    md_info = copy.deepcopy(md_info)
    md_info["markdown_texts"] = transform_markdown_text(
        md_info["markdown_texts"], table_format
    )
    return md_info

def expected_variant_markdown_path(
    pdf_path: Path,
    output_dir: Path,
    variant: str,
) -> Path:
    base_dir = pdf_out_dir(output_dir, pdf_path)
    stem = pdf_path.stem
    if variant == "main":
        return base_dir / f"{stem}.md"
    return base_dir / f"{stem}.{variant}.md"

def save_markdown_variants(
    pipeline: Any,
    output: List[Any],
    input_path: Path,
    out_dir: Path,
    table_format: str,
    chart_output: str,
) -> Dict[str, Any]:
    variants: List[Tuple[str, str, bool]] = [
        ("main", table_format, chart_output == "parsed")
    ]

    all_markdown_images: Dict[str, Any] = {}
    for variant_name, variant_table_format, chart_as_table in variants:
        markdown_list = [
            render_markdown_info(
                res,
                table_format=variant_table_format,
                chart_as_table=chart_as_table,
            )
            for res in output
        ]
        markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

        path_mappings: Dict[str, str] = {}
        for md_info in markdown_list:
            images_dict = md_info.get("markdown_images", {})
            all_markdown_images.update(images_dict)
            for old_path in images_dict.keys():
                path_mappings[old_path] = str(Path("img") / Path(old_path).name)

        for old_path, new_path in path_mappings.items():
            markdown_texts = markdown_texts.replace(old_path, new_path)

        mkd_path = expected_variant_markdown_path(input_path, out_dir, variant_name)
        mkd_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mkd_path, "w", encoding="utf-8") as f:
            f.write(markdown_texts)

    return all_markdown_images

def _init_pipeline(
    vllm_url: str,
    child_timeout_s: int,
    table_format: str,
    chart_output: str,
) -> None:
    global _PIPELINE, _CHILD_TIMEOUT_S, _TABLE_FORMAT, _CHART_OUTPUT
    PaddleOCRVL = import_paddleocr_vl()  # 延迟导入避免主进程继承原生状态
    _PIPELINE = PaddleOCRVL(
        pipeline_version="v1.5",
        vl_rec_backend="vllm-server",
        vl_rec_server_url=vllm_url,
        use_chart_recognition=(chart_output != "off"),
    )
    _CHILD_TIMEOUT_S = int(child_timeout_s)
    _TABLE_FORMAT = table_format
    _CHART_OUTPUT = chart_output

def _get_pipeline() -> Any:
    assert _PIPELINE is not None, "Pipeline not initialized in this process."
    return _PIPELINE

def _with_timeout_posix(seconds: int):
    class _Timeout:
        def __enter__(self):
            if seconds > 0 and hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, self._raise)
                signal.alarm(seconds)
        def __exit__(self, exc_type, exc, tb):
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        @staticmethod
        def _raise(signum, frame):
            raise TimeoutError(f"Worker timed out after {seconds}s")
    return _Timeout()

def parse_pdf_to_markdown(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    pipeline: Any,
    table_format: str,
    chart_output: str,
) -> Path:
    input_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = pdf_out_dir(out_dir, input_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    output = pipeline.predict(input=str(input_path))
    markdown_images = save_markdown_variants(
        pipeline=pipeline,
        output=output,
        input_path=input_path,
        out_dir=out_dir,
        table_format=table_format,
        chart_output=chart_output,
    )
    mkd_final = expected_variant_markdown_path(input_path, out_dir, "main")

    for old, image in markdown_images.items():
        dst_rel = str(Path("img") / Path(old).name)
        file_path = pdf_dir / dst_rel
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(file_path)      # 依赖扩展名推断格式
        try:
            image.close()
        except Exception:
            pass

    return mkd_final

def _worker(pdf_path: Path, out_dir: Path) -> Tuple[str, float, str]:
    start = time.perf_counter()
    try:
        with _with_timeout_posix(_CHILD_TIMEOUT_S):
            pipe = _get_pipeline()
            parse_pdf_to_markdown(
                pdf_path,
                out_dir,
                pipe,
                table_format=_TABLE_FORMAT,
                chart_output=_CHART_OUTPUT,
            )
        dt = time.perf_counter() - start
        return (str(pdf_path), dt, "")
    except BaseException as e:
        dt = time.perf_counter() - start
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return (str(pdf_path), dt, f"{type(e).__name__}: {tb}")

# =======================
# 批处理调度
# =======================

def refresh_manifest_from_fs(conn: sqlite3.Connection, pdfs: List[Path]) -> None:
    with conn:
        for p in pdfs:
            manifest_upsert_file(conn, p)

def process_in_batches(
    conn: sqlite3.Connection,
    pdf_list: List[Path],
    out_dir: Path,
    vllm_url: str,
    jobs: int,
    timeout_s: int,
    max_attempts: int,
    batch_size: int,
    shuffle: bool,
    table_format: str,
    chart_output: str,
) -> None:
    total = len(pdf_list)
    print(f"Discovered {total} PDF(s). Manifest initialized.")

    while True:
        pending = manifest_select_pending(conn, max_attempts=max_attempts, limit=batch_size)
        if not pending:
            break
        if shuffle:
            random.shuffle(pending)

        manifest_bulk_mark_running(conn, pending)
        print(f"Starting batch with {len(pending)} task(s). Spawning pool with {jobs} workers.")
        wall_t0 = time.perf_counter()

        ok_items: List[Tuple[str, float]] = []
        fail_items: List[Tuple[str, float, str]] = []

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=ctx,
            initializer=_init_pipeline,
            initargs=(vllm_url, timeout_s, table_format, chart_output),
        ) as ex:
            futures = {ex.submit(_worker, Path(p), out_dir): p for p in pending}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    path_s, dt, err = fut.result()
                    pdf_path = Path(path_s)
                except BaseException as e:
                    dt = 0.0
                    err = f"ExecutorError: {type(e).__name__}: {e}"
                    pdf_path = Path(p)

                if not err and output_is_valid(pdf_path, out_dir):
                    ok_items.append((str(pdf_path), dt))
                    print(f"[OK] {pdf_path} in {dt:.2f}s")
                else:
                    fail_items.append((str(pdf_path), dt, err or "InvalidOutput"))
                    print(f"[FAIL] {pdf_path}: {err}")

        # 批量提交结果，减少 HDD 同步压力
        manifest_bulk_commit_results(conn, ok_items, fail_items)
        pending_left = len(manifest_select_pending(conn, max_attempts=max_attempts))
        wall_dt = time.perf_counter() - wall_t0
        print(f"Batch done in {wall_dt:.2f}s. Progress: +{len(ok_items)} ok, +{len(fail_items)} fail, pending={pending_left}.")

    print("All batches complete.")
    remain = len(manifest_select_pending(conn, max_attempts=max_attempts))
    if remain > 0:
        print(f"{remain} file(s) still pending after reaching max attempts).")

# =======================
# CLI
# =======================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF-to-Markdown with PaddleOCRVL remote VLM server mode. Run locally and point to a remote /v1 endpoint."
    )
    parser.add_argument("input", help="Path to a PDF file or a directory of PDFs.")
    parser.add_argument("-o", "--output", default="output", help="Output directory.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("-v", "--vllm-url", default="http://127.0.0.1:8118/v1", help="Remote VLM server URL. Host:port is accepted and /v1 will be appended automatically.")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Number of parallel workers (processes).")
    parser.add_argument("--timeout-s", type=int, default=900, help="Per-file hard timeout in seconds.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per file before giving up.")
    parser.add_argument("--batch-size", type=int, default=500, help="Tasks per batch before pool recycle.")
    parser.add_argument("--manifest", default=None, help="Path to manifest SQLite DB, default <output>/manifest.db")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle order inside each batch.")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip files whose outputs already look valid.")
    parser.add_argument("--skip-server-check", action="store_true", help="Skip the remote /health check before starting.")
    parser.add_argument("--table-format", choices=["raw", "clean", "markdown"], default="raw", help="How to render table/chart HTML blocks in Markdown.")
    parser.add_argument("--chart-output", choices=["off", "parsed"], default="off", help="Whether to keep chart blocks as images or parse them into tables.")
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else (output_path / "manifest.db")
    conn = open_manifest(manifest_path)
    manifest_repair_running(conn)
    args.vllm_url = normalize_vllm_url(args.vllm_url)
    ensure_no_proxy_for_vllm_url(args.vllm_url)
    print(f"Using remote VLM server: {args.vllm_url}")
    if not args.skip_server_check:
        check_remote_server(args.vllm_url)
        print("Remote VLM server is reachable.")

    if input_path.is_file():
        # 单文件模式，顺序执行
        PaddleOCRVL = import_paddleocr_vl()
        pipe = PaddleOCRVL(
            pipeline_version="v1.5",
            vl_rec_backend="vllm-server",
            vl_rec_server_url=args.vllm_url,
            use_chart_recognition=(args.chart_output != "off"),
        )
        manifest_upsert_file(conn, input_path)

        if args.skip_if_exists and output_is_valid(input_path, output_path):
            print("Output already exists and looks valid. Skipping.")
            with conn:
                conn.execute(
                    "UPDATE files SET status=?,updated_at=? WHERE path=?;",
                    (STATUS_SUCCESS, time.time(), str(input_path)),
                )
            return

        print(f"Processing file: {input_path}")
        t0 = time.perf_counter()
        try:
            with _with_timeout_posix(args.timeout_s):
                parse_pdf_to_markdown(
                    input_path,
                    output_path,
                    pipe,
                    table_format=args.table_format,
                    chart_output=args.chart_output,
                )
            dt = time.perf_counter() - t0
            if output_is_valid(input_path, output_path):
                manifest_bulk_commit_results(conn, [(str(input_path), dt)], [])
                print(f"Done in {dt:.2f}s")
            else:
                manifest_bulk_commit_results(conn, [], [(str(input_path), dt, "InvalidOutput")])
                print("Parsing finished but output invalid.", file=sys.stderr)
                sys.exit(2)
        except BaseException as e:
            dt = time.perf_counter() - t0
            manifest_bulk_commit_results(conn, [], [(str(input_path), dt, f"{type(e).__name__}: {e}")])
            print(f"Failed: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)

    elif input_path.is_dir():
        pdfs = list_pdfs(input_path, args.recursive)
        if not pdfs:
            print("No PDFs found to process.")
            return
        refresh_manifest_from_fs(conn, pdfs)

        if args.skip_if_exists:
            # 将已有有效输出标记为 success
            ok_paths: List[str] = []
            for p in pdfs:
                if output_is_valid(p, output_path):
                    ok_paths.append(str(p))
            with conn:
                for p in ok_paths:
                    conn.execute(
                        "UPDATE files SET status=?,updated_at=? WHERE path=?;",
                        (STATUS_SUCCESS, time.time(), p),
                    )

        jobs = max(1, args.jobs)
        if jobs == 1:
            PaddleOCRVL = import_paddleocr_vl()
            pipe = PaddleOCRVL(
                pipeline_version="v1.5",
                vl_rec_backend="vllm-server",
                vl_rec_server_url=args.vllm_url,
                use_chart_recognition=(args.chart_output != "off"),
            )
            todo = manifest_select_pending(conn, max_attempts=args.max_attempts)
            for p in todo:
                pdf_path = Path(p)
                t0 = time.perf_counter()
                with conn:
                    conn.execute("UPDATE files SET status=?,updated_at=? WHERE path=?;", (STATUS_RUNNING, time.time(), p))
                try:
                    with _with_timeout_posix(args.timeout_s):
                        parse_pdf_to_markdown(
                            pdf_path,
                            output_path,
                            pipe,
                            table_format=args.table_format,
                            chart_output=args.chart_output,
                        )
                    dt = time.perf_counter() - t0
                    if output_is_valid(pdf_path, output_path):
                        manifest_bulk_commit_results(conn, [(p, dt)], [])
                        print(f"[OK] {p} in {dt:.2f}s")
                    else:
                        manifest_bulk_commit_results(conn, [], [(p, dt, "InvalidOutput")])
                        print(f"[FAIL] {p}: InvalidOutput")
                except BaseException as e:
                    dt = time.perf_counter() - t0
                    manifest_bulk_commit_results(conn, [], [(p, dt, f"{type(e).__name__}: {e}")])
                    print(f"[FAIL] {p}: {type(e).__name__}: {e}")
        else:
            process_in_batches(
                conn=conn,
                pdf_list=pdfs,
                out_dir=output_path,
                vllm_url=args.vllm_url,
                jobs=jobs,
                timeout_s=args.timeout_s,
                max_attempts=args.max_attempts,
                batch_size=max(1, args.batch_size),
                shuffle=args.shuffle,
                table_format=args.table_format,
                chart_output=args.chart_output,
            )
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

if __name__ == "__main__":
    try:
        time_start = time.time()
        main()
        time_end = time.time()
        print(f"Total Time taken: {time_end - time_start} seconds")
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
