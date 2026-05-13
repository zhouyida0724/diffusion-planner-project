#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path

def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def main():
    state_path = Path("/home/zhouyida/.openclaw/workspace/memory/fast_eval_watch.json")
    state = read_json(state_path, {}) or {}
    jsonl_path = Path(state.get("fast_eval_jsonl", ""))
    last = int(state.get("last_reported_step", -1))

    exp_dir = Path(state.get("exp_dir", "")) if state.get("exp_dir") else None
    log_path = (exp_dir / "fast_eval_macro_cron.log") if exp_dir else Path("/tmp/fast_eval_macro_cron.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip("\n") + "\n")

    if not jsonl_path.is_file():
        log(f"no_jsonl path={jsonl_path}")
        return 0

    new = []  # list of dicts to report
    max_step = last
    latest_seen = -1
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("city") != "macro":
                continue
            step = int(o.get("step", -1))
            if step > latest_seen:
                latest_seen = step
            if step <= last:
                continue
            max_step = max(max_step, step)
            new.append(o)

    if not new:
        log(f"no_new last={last} latest_seen={latest_seen}")
        return 0

    new.sort(key=lambda x: int(x.get("step", -1)))

    lines = ["fast-eval update (macro)"]
    for o in new:
        def f3(x):
            try:
                return f"{float(x):.3f}"
            except Exception:
                return "nan"
        step = int(o.get("step", -1))
        loss = f3(o.get("val_loss_proxy"))
        ade = "/".join([f3(o.get("ade_1s")), f3(o.get("ade_3s")), f3(o.get("ade_5s")), f3(o.get("ade_8s"))])
        fde = "/".join([f3(o.get("fde_1s")), f3(o.get("fde_3s")), f3(o.get("fde_5s")), f3(o.get("fde_8s"))])
        lines.append(f"- step {step} | loss_proxy {loss} | ADE(1/3/5/8)={ade} | FDE(1/3/5/8)={fde}")

    msg = "\n".join(lines)

    # Send to Discord DM via OpenClaw.
    # IMPORTANT: do not suppress stderr; cron env problems must be visible.
    openclaw_bin = os.environ.get("OPENCLAW_BIN", "") or "/home/zhouyida/.npm-global/bin/openclaw"
    cmd = [
        openclaw_bin,
        "message",
        "send",
        "--channel",
        "discord",
        "--target",
        "user:445781625561677826",
        "--message",
        msg,
    ]

    sent_ok = False
    try:
        cp = subprocess.run(cmd, check=False, text=True, capture_output=True, env=os.environ.copy())
        sent_ok = (cp.returncode == 0)
        log(f"send_attempt max_step={max_step} count={len(new)} returncode={cp.returncode}")
        if cp.stdout:
            log("--- stdout ---\n" + cp.stdout)
        if cp.stderr:
            log("--- stderr ---\n" + cp.stderr)
        log("====")
    except Exception as e:
        log(f"send_exception={e!r}\n====")

    # Only advance watermark when send succeeded.
    if sent_ok:
        state["last_reported_step"] = int(max_step)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
