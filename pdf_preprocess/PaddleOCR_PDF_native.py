from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Tuple

from PaddleOCR_PDF_share import (
    STATUS_RUNNING,
    import_paddleocr_vl,
    list_pdfs,
    manifest_bulk_commit_results,
    manifest_bulk_mark_running,
    manifest_repair_running,
    manifest_select_pending,
    manifest_upsert_file,
    open_manifest,
    output_is_valid,
    parse_pdf_to_markdown,
)


_PIPELINE = None
_CHILD_TIMEOUT_S = 0
_TABLE_FORMAT = "raw"
_CHART_OUTPUT = "off"


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


def _init_pipeline(
    child_timeout_s: int,
    table_format: str,
    chart_output: str,
    pipeline_version: str,
    vl_rec_model_name: str | None,
    vl_rec_model_dir: str | None,
    disable_model_source_check: bool,
) -> None:
    global _PIPELINE, _CHILD_TIMEOUT_S, _TABLE_FORMAT, _CHART_OUTPUT
    if disable_model_source_check:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    PaddleOCRVL = import_paddleocr_vl()
    kwargs = {
        "pipeline_version": pipeline_version,
        "vl_rec_backend": "native",
        "use_chart_recognition": (chart_output != "off"),
    }
    if vl_rec_model_name:
        kwargs["vl_rec_model_name"] = vl_rec_model_name
    if vl_rec_model_dir:
        kwargs["vl_rec_model_dir"] = vl_rec_model_dir

    _PIPELINE = PaddleOCRVL(**kwargs)
    _CHILD_TIMEOUT_S = int(child_timeout_s)
    _TABLE_FORMAT = table_format
    _CHART_OUTPUT = chart_output


def _get_pipeline() -> Any:
    assert _PIPELINE is not None, "Pipeline not initialized in this process."
    return _PIPELINE


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


def refresh_manifest_from_fs(conn, pdfs: List[Path]) -> None:
    with conn:
        for p in pdfs:
            manifest_upsert_file(conn, p)


def process_in_batches(
    conn,
    pdf_list: List[Path],
    out_dir: Path,
    jobs: int,
    timeout_s: int,
    max_attempts: int,
    batch_size: int,
    shuffle: bool,
    table_format: str,
    chart_output: str,
    pipeline_version: str,
    vl_rec_model_name: str | None,
    vl_rec_model_dir: str | None,
    disable_model_source_check: bool,
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
        print(f"Starting batch with {len(pending)} task(s). Spawning pool with {jobs} worker(s).")
        wall_t0 = time.perf_counter()

        ok_items: List[Tuple[str, float]] = []
        fail_items: List[Tuple[str, float, str]] = []

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=ctx,
            initializer=_init_pipeline,
            initargs=(
                timeout_s,
                table_format,
                chart_output,
                pipeline_version,
                vl_rec_model_name,
                vl_rec_model_dir,
                disable_model_source_check,
            ),
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
                    print(f"[FAIL] {pdf_path}: {err or 'InvalidOutput'}")

        manifest_bulk_commit_results(conn, ok_items, fail_items)
        pending_left = len(manifest_select_pending(conn, max_attempts=max_attempts))
        wall_dt = time.perf_counter() - wall_t0
        print(
            f"Batch done in {wall_dt:.2f}s. Progress: +{len(ok_items)} ok, "
            f"+{len(fail_items)} fail, pending={pending_left}."
        )

    print("All batches complete.")
    remain = len(manifest_select_pending(conn, max_attempts=max_attempts))
    if remain > 0:
        print(f"{remain} file(s) still pending after reaching max attempts).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PDF-to-Markdown with PaddleOCRVL native mode. "
            "Runs local standalone inference without a vLLM server."
        )
    )
    parser.add_argument("input", help="Path to a PDF file or a directory of PDFs.")
    parser.add_argument("-o", "--output", default="output", help="Output directory.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers. Native mode loads one full model per worker; keep this low.",
    )
    parser.add_argument("--timeout-s", type=int, default=1800, help="Per-file hard timeout in seconds.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per file before giving up.")
    parser.add_argument("--batch-size", type=int, default=64, help="Tasks per batch before pool recycle.")
    parser.add_argument("--manifest", default=None, help="Path to manifest SQLite DB, default <output>/manifest.db")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle order inside each batch.")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip files whose outputs already look valid.")
    parser.add_argument(
        "--table-format",
        choices=["raw", "clean", "markdown"],
        default="raw",
        help="How to render table/chart HTML blocks in Markdown.",
    )
    parser.add_argument(
        "--chart-output",
        choices=["off", "parsed"],
        default="off",
        help="Whether to parse chart blocks into tables. Regular extracted images are still saved under img/.",
    )
    parser.add_argument(
        "--pipeline-version",
        choices=["v1", "v1.5"],
        default="v1.5",
        help="PaddleOCRVL pipeline version.",
    )
    parser.add_argument(
        "--vl-rec-model-name",
        default=None,
        help="Optional native VL model name. Leave empty to use the official default model.",
    )
    parser.add_argument(
        "--vl-rec-model-dir",
        default=None,
        help="Optional local path to a downloaded VL model directory.",
    )
    parser.add_argument(
        "--disable-model-source-check",
        action="store_true",
        help="Set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True before model initialization.",
    )
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

    if args.jobs > 1:
        print(
            "Warning: native mode with jobs > 1 will load one full model per worker. "
            "Reduce --jobs to 1 if you hit OOM or repeated worker crashes.",
            file=sys.stderr,
        )

    if input_path.is_file():
        _init_pipeline(
            child_timeout_s=args.timeout_s,
            table_format=args.table_format,
            chart_output=args.chart_output,
            pipeline_version=args.pipeline_version,
            vl_rec_model_name=args.vl_rec_model_name,
            vl_rec_model_dir=args.vl_rec_model_dir,
            disable_model_source_check=args.disable_model_source_check,
        )
        pipe = _get_pipeline()
        manifest_upsert_file(conn, input_path)

        if args.skip_if_exists and output_is_valid(input_path, output_path):
            print("Output already exists and looks valid. Skipping.")
            with conn:
                conn.execute(
                    "UPDATE files SET status=?,updated_at=? WHERE path=?;",
                    ("success", time.time(), str(input_path)),
                )
            return

        print(f"Processing file: {input_path}")
        t0 = time.perf_counter()
        try:
            with _with_timeout_posix(args.timeout_s):
                mkd_path = parse_pdf_to_markdown(
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
                print(f"Markdown: {mkd_path}")
                print(f"Images: {mkd_path.parent / 'img'}")
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
            ok_paths: List[str] = []
            for p in pdfs:
                if output_is_valid(p, output_path):
                    ok_paths.append(str(p))
            with conn:
                for p in ok_paths:
                    conn.execute(
                        "UPDATE files SET status=?,updated_at=? WHERE path=?;",
                        ("success", time.time(), p),
                    )

        jobs = max(1, args.jobs)
        if jobs == 1:
            _init_pipeline(
                child_timeout_s=args.timeout_s,
                table_format=args.table_format,
                chart_output=args.chart_output,
                pipeline_version=args.pipeline_version,
                vl_rec_model_name=args.vl_rec_model_name,
                vl_rec_model_dir=args.vl_rec_model_dir,
                disable_model_source_check=args.disable_model_source_check,
            )
            pipe = _get_pipeline()
            todo = manifest_select_pending(conn, max_attempts=args.max_attempts)
            for p in todo:
                pdf_path = Path(p)
                t0 = time.perf_counter()
                with conn:
                    conn.execute(
                        "UPDATE files SET status=?,updated_at=? WHERE path=?;",
                        (STATUS_RUNNING, time.time(), p),
                    )
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
                jobs=jobs,
                timeout_s=args.timeout_s,
                max_attempts=args.max_attempts,
                batch_size=max(1, args.batch_size),
                shuffle=args.shuffle,
                table_format=args.table_format,
                chart_output=args.chart_output,
                pipeline_version=args.pipeline_version,
                vl_rec_model_name=args.vl_rec_model_name,
                vl_rec_model_dir=args.vl_rec_model_dir,
                disable_model_source_check=args.disable_model_source_check,
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
