"""Backfill Garmin Connect data for periods the live BLE stream missed.

This is analytics-grade backfill from Garmin Cloud — *not* a substitute for
the 1 Hz live BLE Heart Rate stream that lives in `hr_logs/`. Daily HR
samples are roughly every 2 minutes; activity HR is per-second only when
the watch was actively recording an activity.

Per-day output goes to `garmin_backfill/YYYY-MM-DD.json` with these keys
when available: heart_rates, sleep, stress, hrv, respiration, body_battery,
spo2, user_summary. Missing data sets are recorded as null with the error
type so you can tell "no data on Garmin Cloud yet" from "API call failed."

Auth: first interactive run logs in with email + password (and prompts for
MFA if your account has it). Tokens are cached under `.garmin_tokens/` so
subsequent runs are non-interactive and safe to schedule via cron.

Usage:
    # First-run interactive login + grab yesterday
    python scripts/garmin_backfill.py

    # Specific date
    python scripts/garmin_backfill.py --date 2026-04-30

    # Range (inclusive both ends)
    python scripts/garmin_backfill.py --start 2026-04-20 --end 2026-04-30

    # Force re-fetch even if a file already exists
    python scripts/garmin_backfill.py --date 2026-04-30 --force

Env vars (in .env or shell):
    GARMIN_USERNAME=you@example.com
    GARMIN_PASSWORD=...

Requires `garminconnect` and `python-dotenv`. See requirements.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

try:
    from garminconnect import Garmin, GarminConnectTooManyRequestsError
except ImportError:
    sys.exit(
        "garminconnect is not installed. Run: pip install garminconnect python-dotenv"
    )

BASE_DIR = Path(__file__).resolve().parent.parent
BACKFILL_DIR = BASE_DIR / "garmin_backfill"
TOKEN_DIR = BASE_DIR / ".garmin_tokens"


def _prompt_mfa() -> str:
    if not sys.stdin.isatty():
        sys.exit(
            "Garmin requires MFA but no TTY available. "
            "Run this manually once (ssh -t) to seed .garmin_tokens/."
        )
    return input("Garmin MFA code (authenticator app or SMS): ").strip()


def login() -> Garmin:
    """Reuse cached tokens if present, otherwise interactive login + cache.

    garminconnect 0.3.x handles persistence natively when `tokenstore` is
    passed to `login()`: it dumps to `garmin_tokens.json` after a credential
    login and re-dumps after each DI token refresh. No manual save needed.
    """
    load_dotenv(BASE_DIR / ".env")
    TOKEN_DIR.mkdir(exist_ok=True)

    # Try cached token first — no creds, no MFA prompt.
    try:
        client = Garmin()
        client.login(str(TOKEN_DIR))
        _assert_login_complete(client)
        return client
    except SystemExit:
        raise
    except Exception:
        pass

    email = os.getenv("GARMIN_USERNAME")
    password = os.getenv("GARMIN_PASSWORD")
    if not email or not password:
        sys.exit(
            "No cached token and GARMIN_USERNAME/GARMIN_PASSWORD not set. "
            "Add them to .env and run interactively once to seed tokens."
        )

    # `prompt_mfa` (not `return_on_mfa`) is required: with return_on_mfa=True
    # the library returns early without loading the user profile, so
    # `display_name` stays None even on a clean no-MFA success — every
    # subsequent endpoint that templates display_name into its URL then 403s.
    client = Garmin(email=email, password=password, prompt_mfa=_prompt_mfa)
    client.login(str(TOKEN_DIR))
    _assert_login_complete(client)
    return client


def _assert_login_complete(client: Garmin) -> None:
    """Fail fast if profile fetch silently failed.

    Garmin's auth flow can issue OAuth tokens but reject the subsequent
    profile fetch under rate-limiting, leaving `display_name` as `None`.
    Several endpoints template `display_name` into the URL, so they 403
    on `/None`. Better to abort here than waste 8 API calls.
    """
    display_name = getattr(client, "display_name", None)
    if not display_name:
        sys.exit(
            "Login returned tokens but profile fetch failed "
            "(display_name is None). Almost always Garmin rate-limiting. "
            "Wait several hours, then retry — ideally from a different IP "
            "for the first successful seed."
        )


class _RateLimited(Exception):
    """Raised when a fetcher hits 429 — caller decides how long to back off."""


def _safe(fn: Callable[[], Any]) -> dict:
    """Run fetcher, return {ok, value|error}. Re-raises 429 for caller backoff."""
    try:
        return {"ok": True, "value": fn()}
    except GarminConnectTooManyRequestsError as e:
        raise _RateLimited(str(e)) from e
    except Exception as e:
        msg = str(e)
        if "429" in msg or "Too Many Requests" in msg:
            raise _RateLimited(msg) from e
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


_DATA_KEYS = (
    "heart_rates", "sleep", "stress", "hrv",
    "respiration", "body_battery", "spo2", "user_summary",
)


def _is_empty(payload: dict) -> bool:
    """Treat a day as empty if HR recorded zero samples.

    Garmin returns shell payloads (non-None dicts with empty arrays) for days
    the watch wasn't worn or hadn't synced yet, so a top-level None check
    misses them. HR sample count is the primary signal for the seizure
    pipeline, so it's the right gate for "should we retry this day."
    """
    hr = payload.get("heart_rates")
    if not isinstance(hr, dict):
        return True
    values = hr.get("heartRateValues")
    return not (isinstance(values, list) and len(values) > 0)


def _empty_file_is_stale(
    path: Path,
    retry_window_days: int,
    min_age_hours: float = 20.0,
) -> bool:
    """Return True iff `path` is empty, within retry window, and last attempt is old.

    The min_age_hours floor stops the daily cron from re-hitting the same
    empty day on every invocation — Garmin Cloud needs hours, not minutes,
    to catch up on a late phone sync.
    """
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    if not _is_empty(payload):
        return False
    try:
        day = date.fromisoformat(payload.get("date") or path.stem)
    except ValueError:
        return False
    if day < date.today() - timedelta(days=retry_window_days):
        return False
    fetched_at = payload.get("fetched_at")
    if not fetched_at:
        return True
    try:
        ts = datetime.fromisoformat(fetched_at)
    except ValueError:
        return True
    return datetime.now(timezone.utc) - ts >= timedelta(hours=min_age_hours)


def collect_stale_empty_dates(retry_window_days: int) -> list[date]:
    if retry_window_days <= 0 or not BACKFILL_DIR.exists():
        return []
    out: list[date] = []
    for f in BACKFILL_DIR.glob("*.json"):
        try:
            day = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if _empty_file_is_stale(f, retry_window_days):
            out.append(day)
    return sorted(out)


def fetch_day(client: Garmin, day: date) -> dict:
    iso = day.isoformat()
    payload = {
        "date": iso,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    fetchers: list[tuple[str, Callable[[], Any]]] = [
        ("heart_rates", lambda: client.get_heart_rates(iso)),
        ("sleep", lambda: client.get_sleep_data(iso)),
        ("stress", lambda: client.get_stress_data(iso)),
        ("hrv", lambda: client.get_hrv_data(iso)),
        ("respiration", lambda: client.get_respiration_data(iso)),
        ("body_battery", lambda: client.get_body_battery(iso, iso)),
        ("spo2", lambda: client.get_spo2_data(iso)),
        ("user_summary", lambda: client.get_user_summary(iso)),
    ]
    for key, fn in fetchers:
        result = _safe(fn)
        payload[key] = result["value"] if result["ok"] else None
        payload[f"{key}_error"] = None if result["ok"] else result["error"]
    return payload


def write_day(payload: dict, force: bool) -> Path | None:
    BACKFILL_DIR.mkdir(exist_ok=True)
    out = BACKFILL_DIR / f"{payload['date']}.json"
    if out.exists() and not force:
        return None
    with out.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--date", help="Single ISO date, e.g. 2026-04-30")
    p.add_argument("--start", help="Range start, ISO date")
    p.add_argument("--end", help="Range end, ISO date (inclusive)")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing backfill files",
    )
    p.add_argument(
        "--per-day-sleep",
        type=float,
        default=2.0,
        help="Base seconds to sleep between days (jittered ±50%%); default 2.0",
    )
    p.add_argument(
        "--cooldown-sec",
        type=int,
        default=300,
        help="Seconds to pause when a 429 is hit before retrying that day; default 300",
    )
    p.add_argument(
        "--retry-empty-within-days",
        type=int,
        default=30,
        help=(
            "On default (no explicit date) runs, also re-fetch existing empty "
            "files within N days whose last attempt was >20h ago; default 30, "
            "set 0 to disable"
        ),
    )
    return p.parse_args()


def resolve_dates(args: argparse.Namespace) -> list[date]:
    if args.date:
        return [date.fromisoformat(args.date)]
    if args.start and args.end:
        s = date.fromisoformat(args.start)
        e = date.fromisoformat(args.end)
        return [s + timedelta(days=i) for i in range((e - s).days + 1)]
    # Default: yesterday. Today's data may not have synced yet, and sleep
    # data for "last night" is keyed under yesterday's date in Garmin's API.
    return [date.today() - timedelta(days=1)]


def main() -> int:
    args = parse_args()
    days = resolve_dates(args)

    # Default cron mode (no explicit date/range): also enqueue stale-empty
    # days within the retry window, so days that synced late get backfilled.
    if not (args.date or args.start or args.end):
        retry_days = collect_stale_empty_dates(args.retry_empty_within_days)
        if retry_days:
            days = sorted(set(days) | set(retry_days))
            print(
                f"queued {len(retry_days)} stale-empty day(s) within last "
                f"{args.retry_empty_within_days} days for re-fetch"
            )

    client = login()
    total = len(days)

    for i, day in enumerate(days, start=1):
        out_path = BACKFILL_DIR / f"{day.isoformat()}.json"
        if out_path.exists() and not args.force:
            # Re-fetch only if the file is empty AND within window AND last
            # attempt is stale. Otherwise skip — saves 8 API calls per day.
            if not _empty_file_is_stale(out_path, args.retry_empty_within_days):
                print(f"[{i}/{total}] [{day}] skipped (file exists)")
                continue
            print(f"[{i}/{total}] [{day}] empty + stale, re-fetching")

        try:
            payload = fetch_day(client, day)
        except _RateLimited as e:
            print(
                f"[{i}/{total}] [{day}] 429 hit ({e}); "
                f"sleeping {args.cooldown_sec}s then retrying once",
                file=sys.stderr,
            )
            time.sleep(args.cooldown_sec)
            try:
                payload = fetch_day(client, day)
            except _RateLimited as e2:
                print(
                    f"[{i}/{total}] [{day}] 429 again after cooldown ({e2}); "
                    "skipping this day — re-run later to fill gap",
                    file=sys.stderr,
                )
                continue

        # Always write (force=True): we may be overwriting an existing empty
        # file with a fresh attempt, regardless of whether new data arrived.
        out = write_day(payload, force=True)
        present = sum(
            1 for k in _DATA_KEYS if payload.get(k) is not None
        )
        empty_tag = " (still empty)" if _is_empty(payload) else ""
        print(
            f"[{i}/{total}] [{day}] wrote {out.name}  "
            f"datasets_present={present}/8{empty_tag}"
        )

        # Throttle between days. Skip the sleep on the last day.
        if i < total and args.per_day_sleep > 0:
            jitter = args.per_day_sleep * random.uniform(0.5, 1.5)
            time.sleep(jitter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
