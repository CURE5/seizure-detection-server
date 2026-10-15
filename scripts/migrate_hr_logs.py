"""Migrate legacy `data_logs/hr_YYYY-MM-DD.jsonl` (one record per line) to
the minute-batched form in `hr_logs/YYYY-MM-DD.jsonl` written by HRLogger.

Idempotent: if the destination file already exists, it is left alone unless
--force is passed. The legacy file is left in place; rename or delete it
manually after verifying the migration.

Usage:
    python scripts/migrate_hr_logs.py [--src DIR] [--dst DIR] [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

LEGACY_RE = re.compile(r"^hr_(\d{4}-\d{2}-\d{2})\.jsonl$")


def _ms_from_midnight(ts: datetime) -> int:
    return (
        ((ts.hour * 60 + ts.minute) * 60 + ts.second) * 1000
        + ts.microsecond // 1000
    )


def migrate_file(src: Path, dst: Path) -> int:
    """Convert one legacy file. Returns number of minute-lines written."""
    minutes: dict[int, dict] = {}
    skipped = 0
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts = datetime.fromisoformat(rec["ts"])
                bpm = rec["bpm"]
            except Exception:
                skipped += 1
                continue
            ts_utc = ts.astimezone(ts.tzinfo) if ts.tzinfo else ts
            minute = ts_utc.hour * 60 + ts_utc.minute
            buf = minutes.setdefault(minute, {
                "min": minute, "t": [], "bpm": [], "rr": [], "contact": [],
            })
            buf["t"].append(_ms_from_midnight(ts_utc))
            buf["bpm"].append(bpm)
            buf["rr"].append(
                [round(x, 1) for x in rec.get("rr_intervals_ms") or []]
            )
            buf["contact"].append(rec.get("sensor_contact"))

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for minute in sorted(minutes):
            f.write(json.dumps(minutes[minute], separators=(",", ":")) + "\n")

    if skipped:
        print(f"  warning: skipped {skipped} unparseable line(s)", file=sys.stderr)
    return len(minutes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data_logs"))
    ap.add_argument("--dst", type=Path, default=Path("hr_logs"))
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing destination files")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"src dir not found: {args.src}", file=sys.stderr)
        return 1
    args.dst.mkdir(parents=True, exist_ok=True)

    files = []
    for p in sorted(args.src.iterdir()):
        m = LEGACY_RE.match(p.name)
        if m:
            files.append((m.group(1), p))

    if not files:
        print(f"No legacy hr_*.jsonl files in {args.src}")
        return 0

    total_lines = 0
    for ymd, src_path in files:
        dst_path = args.dst / f"{ymd}.jsonl"
        if dst_path.exists() and not args.force:
            print(f"skip  {dst_path}  (exists; use --force to overwrite)")
            continue
        n = migrate_file(src_path, dst_path)
        total_lines += n
        print(f"wrote {dst_path}  ({n} minute-lines from {src_path.name})")

    print(f"\nDone. {total_lines} minute-lines written across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
