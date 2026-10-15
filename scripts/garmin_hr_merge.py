"""Merge Garmin Cloud HR backfill into `hr_logs/` where the live BLE feed has gaps.

Reads:  `garmin_backfill/YYYY-MM-DD.json` (the `heart_rates` payload)
Writes: appends to `hr_logs/YYYY-MM-DD.jsonl`, only for minutes the live feed
        didn't already cover.

Each backfilled line carries `"source": "garmin_backfill"` so downstream code
can distinguish ~2-min-resolution cloud samples from 1 Hz BLE-live samples.
**Backfill samples must not feed the seizure-detection signal path** — they're
too sparse to model tachycardia/bradycardia onset reliably; they're for
analytics-grade backfill only (sleep correlation, daily summaries, etc.).

Idempotent: re-running won't duplicate. Any minute already present in the
log file (whether from live capture or a prior backfill run) is skipped.

Usage:
    python scripts/garmin_hr_merge.py --date 2026-04-30
    python scripts/garmin_hr_merge.py --start 2026-04-25 --end 2026-04-30
    python scripts/garmin_hr_merge.py --dry-run                  # default: yesterday
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKFILL_DIR = BASE_DIR / "garmin_backfill"
HR_LOGS_DIR = BASE_DIR / "hr_logs"


def load_existing_minutes(jsonl_path: Path) -> set[int]:
    """Return the set of minute-of-day ints already present in the log file.

    Treats malformed lines as missing rather than fatal — append-only logs in
    the wild occasionally have a torn final line if the writer crashed mid-flush.
    """
    if not jsonl_path.exists():
        return set()
    minutes: set[int] = set()
    with jsonl_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and "min" in rec:
                    minutes.add(int(rec["min"]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return minutes


def heart_rate_samples(backfill_payload: dict) -> list[tuple[int, int]]:
    """Pull `(epoch_ms, bpm)` tuples from the backfill JSON, skipping nulls.

    Garmin's `heartRateValues` is a list of `[epoch_ms, bpm_or_null]`; null
    bpm means "no measurement at this slot" — those are dropped.
    """
    hr = backfill_payload.get("heart_rates")
    if not hr:
        return []
    raw = hr.get("heartRateValues") or []
    out: list[tuple[int, int]] = []
    for entry in raw:
        try:
            ts_ms, bpm = entry
        except (TypeError, ValueError):
            continue
        if bpm is None:
            continue
        out.append((int(ts_ms), int(bpm)))
    return out


def epoch_ms_to_minute_of_day_utc(ts_ms: int) -> tuple[int, int]:
    """Convert epoch ms → (minute-of-day UTC, ms-since-midnight UTC).

    Matches the convention in `hr_logs/*.jsonl`: `min` is minute of UTC day,
    `t` entries are ms since UTC midnight (verified: `t / 60000 == min`).
    """
    utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    midnight = utc.replace(hour=0, minute=0, second=0, microsecond=0)
    ms_since_midnight = int((utc - midnight).total_seconds() * 1000)
    return ms_since_midnight // 60000, ms_since_midnight


def merge_one_day(date_str: str, dry_run: bool = False) -> None:
    backfill_path = BACKFILL_DIR / f"{date_str}.json"
    if not backfill_path.exists():
        print(f"[{date_str}] no backfill file at {backfill_path}; skipping")
        return

    with backfill_path.open() as f:
        payload = json.load(f)

    samples = heart_rate_samples(payload)
    if not samples:
        err = payload.get("heart_rates_error")
        suffix = f" (error: {err})" if err else ""
        print(f"[{date_str}] no HR data in backfill; nothing to merge{suffix}")
        return

    log_path = HR_LOGS_DIR / f"{date_str}.jsonl"
    HR_LOGS_DIR.mkdir(exist_ok=True)
    existing_minutes = load_existing_minutes(log_path)

    # Group remaining (uncovered) Garmin samples by minute-of-day. Multiple
    # Garmin samples in the same minute is rare (sampling is ~2 min apart)
    # but we coalesce defensively rather than write multiple backfill lines
    # for one minute.
    by_minute: dict[int, list[tuple[int, int]]] = {}
    skipped_live = 0
    for ts_ms, bpm in samples:
        minute, ms_since_midnight = epoch_ms_to_minute_of_day_utc(ts_ms)
        if minute in existing_minutes:
            skipped_live += 1
            continue
        by_minute.setdefault(minute, []).append((ms_since_midnight, bpm))

    new_lines: list[str] = []
    for minute in sorted(by_minute):
        ms_list, bpm_list = zip(*by_minute[minute])
        new_lines.append(json.dumps({
            "min": minute,
            "t": list(ms_list),
            "bpm": list(bpm_list),
            "rr": [[] for _ in ms_list],
            "contact": [None for _ in ms_list],
            "source": "garmin_backfill",
        }))

    summary = (
        f"[{date_str}] {len(new_lines)} minutes to backfill, "
        f"skipped {skipped_live} (live coverage)"
    )
    if dry_run:
        print(f"[DRY RUN] {summary}")
        return

    if new_lines:
        with log_path.open("a") as f:
            for line in new_lines:
                f.write(line + "\n")
    print(summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--date", help="Single ISO date, e.g. 2026-04-30")
    p.add_argument("--start", help="Range start, ISO date")
    p.add_argument("--end", help="Range end, ISO date (inclusive)")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be appended without writing",
    )
    return p.parse_args()


def resolve_dates(args: argparse.Namespace) -> list[str]:
    if args.date:
        return [args.date]
    if args.start and args.end:
        s = date.fromisoformat(args.start)
        e = date.fromisoformat(args.end)
        return [
            (s + timedelta(days=i)).isoformat()
            for i in range((e - s).days + 1)
        ]
    # Default: yesterday — same convention as garmin_backfill.py.
    return [(date.today() - timedelta(days=1)).isoformat()]


def main() -> int:
    args = parse_args()
    for d in resolve_dates(args):
        merge_one_day(d, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
