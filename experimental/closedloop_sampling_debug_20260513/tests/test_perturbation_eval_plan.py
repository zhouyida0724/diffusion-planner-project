from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "dp_perturbation_eval_plan.py"


def test_generates_closed_loop_perturbation_matrix_and_runbook(tmp_path: Path) -> None:
    token_csv = tmp_path / "tokens.csv"
    token_csv.write_text("token,split\nabc123,left\n", encoding="utf-8")
    output_root = tmp_path / "outputs" / "闭环 eval root"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--token-csv",
            str(token_csv),
            "--base-output-root",
            str(output_root),
            "--checkpoint",
            "outputs/training/ckpt.pt",
            "--data-root",
            str(tmp_path / "中文 data root"),
            "--sampling-steps",
            "12",
            "--selector-enable",
            "--lateral-m",
            "0.5,1.0",
            "--heading-deg",
            "3,6",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    matrix_path = output_root / "perturbation_run_matrix.csv"
    runbook_path = output_root / "perturbation_runbook.md"
    rows = list(csv.DictReader(matrix_path.open(encoding="utf-8")))

    assert len(rows) == 9
    baseline = [row for row in rows if row["run_id"] == "baseline"]
    assert len(baseline) == 1
    assert baseline[0]["is_baseline"] == "true"
    assert {row["perturbation_kind"] for row in rows} == {"baseline", "lateral", "heading"}
    assert all(row["selector_enable"] == "true" for row in rows)

    runbook = runbook_path.read_text(encoding="utf-8")
    assert "Hydra 中文 data-root quoting 建议" in runbook
    assert "scenario_builder.data_root='/tmp/" in runbook
    assert "当前未发现可用的初始扰动注入接口" in runbook
    assert "DP_INITIAL_PERTURB_LATERAL_M" in runbook
