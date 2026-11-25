"""One-click GitHub Cloud Sync for xau_mtf (pushes code, trained models, v2 reports, and synced Parquet)."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> tuple[int, str]:
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    out = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
    return res.returncode, out


def cloud_sync_now(message: str | None = None) -> tuple[bool, str]:
    if not (ROOT / ".git").exists():
        return (
            False,
            "Git repo not initialized inside xau_mtf/ yet.\n"
            "Run once inside xau_mtf/:\n"
            "  git init\n"
            "  git branch -M main\n"
            "  git remote add origin https://github.com/<YOUR_USER>/xau-ali.git\n"
            "Then double-click cloud_sync_now.bat!",
        )

    code, out = _run(["git", "add", "-A"])
    if code != 0:
        return False, f"git add failed:\n{out}"

    code_st, st_out = _run(["git", "status", "--porcelain"])
    if not st_out.strip():
        return True, "Already up-to-date — nothing new to push."

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = message or f"xau_mtf cloud sync ({ts})"
    code, out = _run(["git", "commit", "-m", msg])
    if code != 0:
        return False, f"git commit failed:\n{out}"

    code, out = _run(["git", "push", "-u", "origin", "main"])
    if code != 0:
        return False, f"git push failed:\n{out}"

    return True, f"Cloud Sync SUCCESS ({ts})!\n{out}"


if __name__ == "__main__":
    ok, info = cloud_sync_now()
    print(info)
    sys.exit(0 if ok else 1)
