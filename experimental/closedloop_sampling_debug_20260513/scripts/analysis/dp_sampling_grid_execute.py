#!/usr/bin/env python3
"""Execute a DP sampling-grid run matrix with resumable records."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = Path("scripts/analysis/dp_selector_trace_diversity_audit.py")
STATUS_FIELDS = [
    "row_index",
    "matrix",
    "run_id",
    "sampling_steps",
    "noise_scale",
    "dpm_variant",
    "status",
    "returncode",
    "started_at",
    "finished_at",
    "duration_s",
    "trace_path",
    "expected_trace_path",
    "trace_exists",
    "trace_bytes",
    "audit_returncode",
    "log_path",
    "command_path",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_status(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STATUS_FIELDS})


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[tuple[int, dict[str, str]]]:
    indexed = list(enumerate(rows))
    if args.only_run_id:
        wanted = set(args.only_run_id)
        indexed = [(index, row) for index, row in indexed if row.get("run_id") in wanted]
    if args.start_index:
        indexed = [(index, row) for index, row in indexed if index >= args.start_index]
    if args.limit is not None:
        indexed = indexed[: args.limit]
    return indexed


def _merge_status_rows(existing_rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    merged_by_index: dict[int, dict[str, Any]] = {}
    for row in existing_rows:
        try:
            row_index = int(row.get("row_index", ""))
        except (TypeError, ValueError):
            continue
        merged_by_index[row_index] = row
    merged_by_index[int(new_row["row_index"])] = new_row
    return [merged_by_index[index] for index in sorted(merged_by_index)]


def _safe_name(row: dict[str, str], row_index: int) -> str:
    return f"{row_index:03d}_{row.get('matrix', 'matrix')}_{row.get('run_id', 'run')}".replace("/", "_")


def _resolve_trace_path(row: dict[str, str]) -> Path:
    """Return expected trace path, or a Hydra-chdir nested trace if it exists."""
    trace_path = Path(row.get("trace_path", ""))
    if trace_path.exists():
        return trace_path
    output_root = Path(row.get("output_root", ""))
    if output_root.exists():
        matches = sorted(
            output_root.glob("**/selector_trace.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    return trace_path


def _trace_info(row: dict[str, str]) -> tuple[Path, bool, int]:
    trace_path = _resolve_trace_path(row)
    if trace_path.exists():
        return trace_path, True, trace_path.stat().st_size
    return trace_path, False, 0


def _run_audit(row: dict[str, str], log_handle: Any) -> int | str:
    trace_path = _resolve_trace_path(row)
    if not trace_path.exists():
        return "missing_trace"
    output_root = Path(row["output_root"])
    audit_cmd = [
        "python3",
        str(AUDIT_SCRIPT),
        "--trace-jsonl",
        str(trace_path),
        "--out-csv",
        str(output_root / "diversity_audit.csv"),
        "--out-md",
        str(output_root / "diversity_audit.md"),
        "--config-id",
        f"{row.get('matrix', '')}/{row.get('run_id', '')}",
        "--run-id",
        row.get("run_id", ""),
    ]
    log_handle.write("\n\n===== selector diversity audit =====\n")
    log_handle.write(" ".join(audit_cmd) + "\n")
    log_handle.flush()
    result = subprocess.run(audit_cmd, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def execute_row(row_index: int, row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    record_name = _safe_name(row, row_index)
    log_dir = args.record_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{record_name}.log"
    command_path = log_dir / f"{record_name}.sh"
    command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + row["command"] + "\n", encoding="utf-8")
    command_path.chmod(0o755)

    started_at = _utc_now()
    started = time.monotonic()
    if args.dry_run:
        returncode = 0
        audit_returncode: int | str = "dry_run"
        log_path.write_text("[dry-run]\n" + row["command"] + "\n", encoding="utf-8")
    elif args.audit_existing:
        returncode = 0
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"started_at={started_at}\n")
            log_handle.write(f"row_index={row_index}\n")
            log_handle.write("[audit-existing]\n")
            log_handle.write(f"expected_trace_path={row.get('trace_path', '')}\n")
            log_handle.write(f"resolved_trace_path={_resolve_trace_path(row)}\n")
            log_handle.flush()
            audit_returncode = _run_audit(row, log_handle) if args.audit else ""
    else:
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"started_at={started_at}\n")
            log_handle.write(f"row_index={row_index}\n")
            log_handle.write(f"command={row['command']}\n\n")
            log_handle.flush()
            result = subprocess.run(
                row["command"],
                shell=True,
                executable=args.shell,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            returncode = result.returncode
            audit_returncode = _run_audit(row, log_handle) if args.audit and returncode == 0 else ""
    finished_at = _utc_now()
    trace_path, trace_exists, trace_bytes = _trace_info(row)
    status = "ok" if returncode == 0 and (audit_returncode in ("", 0, "dry_run")) else "failed"
    return {
        "row_index": row_index,
        "matrix": row.get("matrix", ""),
        "run_id": row.get("run_id", ""),
        "sampling_steps": row.get("sampling_steps", ""),
        "noise_scale": row.get("noise_scale", ""),
        "dpm_variant": row.get("dpm_variant", ""),
        "status": status,
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": f"{time.monotonic() - started:.3f}",
        "trace_path": str(trace_path),
        "expected_trace_path": row.get("trace_path", ""),
        "trace_exists": int(trace_exists),
        "trace_bytes": trace_bytes,
        "audit_returncode": audit_returncode,
        "log_path": str(log_path),
        "command_path": str(command_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", required=True, type=Path)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-run-id", action="append", default=[])
    parser.add_argument("--shell", default="/bin/bash")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Preserve existing status.csv and skip rows already marked ok.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.record_dir.mkdir(parents=True, exist_ok=True)
    status_csv = args.record_dir / "status.csv"
    status_jsonl = args.record_dir / "status.jsonl"
    existing_status_rows = _read_status(status_csv) if args.resume else []
    completed_ok_indices = {
        int(row["row_index"])
        for row in existing_status_rows
        if str(row.get("status", "")) == "ok" and str(row.get("row_index", "")).isdigit()
    }
    rows = [
        (row_index, row)
        for row_index, row in _select_rows(_read_matrix(args.matrix_csv), args)
        if not args.resume or row_index not in completed_ok_indices
    ]
    status_rows: list[dict[str, Any]] = list(existing_status_rows)
    manifest = {
        "matrix_csv": str(args.matrix_csv),
        "record_dir": str(args.record_dir),
        "selected_rows": len(rows),
        "resume": bool(args.resume),
        "skipped_completed_rows": len(completed_ok_indices) if args.resume else 0,
        "audit": bool(args.audit),
        "dry_run": bool(args.dry_run),
        "started_at": _utc_now(),
    }
    (args.record_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for row_index, row in rows:
        status = execute_row(row_index, row, args)
        status_rows = _merge_status_rows(status_rows, status)
        _append_jsonl(status_jsonl, status)
        _write_csv(status_csv, status_rows)
        if status["status"] != "ok" and not args.continue_on_failure:
            return int(status["returncode"]) or 1
    if not rows:
        _write_csv(status_csv, status_rows)
    return 0 if status_rows and all(row["status"] == "ok" for row in status_rows) else (0 if not status_rows else 1)


if __name__ == "__main__":
    raise SystemExit(main())
