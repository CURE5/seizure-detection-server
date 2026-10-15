"""Heart-rate sample logger.

Persists raw HR samples in `hr_logs/YYYY-MM-DD.jsonl`, one line per minute,
parallel arrays inside each line:

    {"min": 515,
     "t":   [31000123, 31001140, ...],   // ms from midnight UTC
     "bpm": [72, 73, ...],
     "rr":  [[1023], [], ...],            // sparse, parallel to bpm
     "contact": [true, true, ...]}        // null when watch doesn't report

Append-only: a minute is buffered in memory and written as one line when the
next minute's first sample arrives, when wall-clock has moved past the buffered
minute (catches the "watch went silent" case), or on explicit flush()
(alert / shutdown).

Derived fields (baseline, above-threshold, etc.) are *not* stored — they're
recomputed at read time from samples + canonical event logs (sz, sleep, ...).
This keeps the on-disk format lossless and decouples it from threshold tuning.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ..config import HR_LOGS_DIR
from ..garmin_hr import HRReading

logger = logging.getLogger(__name__)


@dataclass
class _MinuteBuffer:
    day: date
    minute: int  # 0..1439, minute-of-day UTC
    t: list[int] = field(default_factory=list)
    bpm: list[int] = field(default_factory=list)
    rr: list[list[float]] = field(default_factory=list)
    contact: list[Optional[bool]] = field(default_factory=list)

    def add(
        self,
        ts_ms_from_midnight: int,
        bpm: int,
        rr: list[float],
        contact: Optional[bool],
    ) -> None:
        self.t.append(ts_ms_from_midnight)
        self.bpm.append(bpm)
        # Round RR to 1ms — sub-ms precision is below sensor accuracy and
        # makes the JSON noisy.
        self.rr.append([round(x, 1) for x in rr])
        self.contact.append(contact)

    def to_record(self) -> dict:
        return {
            "min": self.minute,
            "t": self.t,
            "bpm": self.bpm,
            "rr": self.rr,
            "contact": self.contact,
        }


class HRLogger:
    """Append-only minute-batched HR sample writer.

    Concurrency: a single asyncio lock guards the in-memory buffer + file
    append. Writes happen via `loop.run_in_executor` to keep the event loop
    free during disk I/O.
    """

    # How often the stale-flush task wakes up. Bounds maximum lag between a
    # minute's last sample and that minute hitting disk when the watch goes
    # silent partway through a minute.
    STALE_CHECK_INTERVAL_SEC = 30

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self.log_dir = Path(log_dir or HR_LOGS_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._buf: Optional[_MinuteBuffer] = None
        self._lock = asyncio.Lock()
        self._stale_task: Optional[asyncio.Task] = None

    def _path(self, day: date) -> Path:
        return self.log_dir / f"{day.isoformat()}.jsonl"

    @staticmethod
    def _ms_from_midnight(ts: datetime) -> int:
        return (
            ((ts.hour * 60 + ts.minute) * 60 + ts.second) * 1000
            + ts.microsecond // 1000
        )

    @staticmethod
    def _append_line_sync(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(line)

    async def add(self, reading: HRReading) -> None:
        """Buffer a reading. Rolls the buffer to disk on minute/day change."""
        ts = reading.timestamp.astimezone(timezone.utc)
        day = ts.date()
        minute = ts.hour * 60 + ts.minute
        ms = self._ms_from_midnight(ts)

        async with self._lock:
            if self._buf is not None and (
                self._buf.day != day or self._buf.minute != minute
            ):
                await self._flush_locked()
            if self._buf is None:
                self._buf = _MinuteBuffer(day=day, minute=minute)
            self._buf.add(
                ms,
                reading.bpm,
                reading.rr_intervals_ms,
                reading.sensor_contact,
            )

    async def flush(self) -> bool:
        """Force-write the current minute's buffer. Returns True if anything was written."""
        async with self._lock:
            return await self._flush_locked()

    async def _flush_locked(self) -> bool:
        if self._buf is None or not self._buf.bpm:
            self._buf = None
            return False
        path = self._path(self._buf.day)
        line = json.dumps(self._buf.to_record(), separators=(",", ":")) + "\n"
        n = len(self._buf.bpm)
        minute = self._buf.minute
        day = self._buf.day
        self._buf = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._append_line_sync, path, line)
        logger.debug(
            "HR log: flushed %d samples for %s minute=%d", n, day, minute,
        )
        return True

    async def _stale_flush_loop(self) -> None:
        """Flush a buffered minute once wall-clock has moved past it.

        Without this, a minute that received its last sample mid-way then went
        silent would sit in memory indefinitely.
        """
        try:
            while True:
                await asyncio.sleep(self.STALE_CHECK_INTERVAL_SEC)
                async with self._lock:
                    if self._buf is None:
                        continue
                    now = datetime.now(timezone.utc)
                    now_minute = now.hour * 60 + now.minute
                    if (
                        self._buf.day != now.date()
                        or self._buf.minute < now_minute
                    ):
                        try:
                            await self._flush_locked()
                        except Exception:
                            logger.exception("HR stale flush failed")
        except asyncio.CancelledError:
            raise

    async def start(self) -> None:
        if self._stale_task is None:
            self._stale_task = asyncio.create_task(
                self._stale_flush_loop(), name="hr-logger-stale-flush",
            )

    async def stop(self) -> None:
        if self._stale_task is not None:
            self._stale_task.cancel()
            try:
                await self._stale_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stale_task = None
        try:
            await self.flush()
        except Exception:
            logger.exception("Final HR flush failed")

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @staticmethod
    def _day_epoch_ms(day: date) -> int:
        return int(
            datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            .timestamp() * 1000
        )

    def _read_day_sync(self, day: date) -> list[dict]:
        path = self._path(day)
        if not path.exists():
            return []
        out: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("HR log: skipping bad line in %s", path.name)
        return out

    async def read_range(self, start: datetime, end: datetime) -> dict:
        """Return samples in the half-open range [start, end) as columnar arrays.

        Output:
            {"t": [epoch_ms, ...], "bpm": [...], "rr": [...], "contact": [...]}

        Spans day-file boundaries. If a partial buffer for `today` is in memory
        and overlaps the range, it is included so live consumers see freshly
        arrived samples without waiting for a flush.
        """
        if end <= start:
            return {"t": [], "bpm": [], "rr": [], "contact": []}

        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        start_ms = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)

        # Snapshot the in-memory buffer (if any) under the lock so an in-flight
        # flush can't race us.
        async with self._lock:
            mem = self._buf.to_record() if self._buf is not None else None
            mem_day = self._buf.day if self._buf is not None else None

        loop = asyncio.get_running_loop()
        t_out: list[int] = []
        bpm_out: list[int] = []
        rr_out: list[list[float]] = []
        contact_out: list[Optional[bool]] = []

        cur = start_utc.date()
        last = end_utc.date()
        while cur <= last:
            day_epoch = self._day_epoch_ms(cur)
            records = await loop.run_in_executor(
                None, self._read_day_sync, cur,
            )
            if mem is not None and cur == mem_day:
                # Drop any flushed line for this minute (defensive — flush
                # always clears _buf, so the duplication shouldn't occur, but
                # guarding here keeps the invariant if something changes later).
                records = [r for r in records if r["min"] != mem["min"]]
                records.append(mem)
            for rec in records:
                ts_arr = rec.get("t") or []
                bpm_arr = rec.get("bpm") or []
                rr_arr = rec.get("rr") or [[]] * len(bpm_arr)
                cont_arr = rec.get("contact") or [None] * len(bpm_arr)
                for i, ts_ms in enumerate(ts_arr):
                    abs_ms = day_epoch + ts_ms
                    if abs_ms < start_ms or abs_ms >= end_ms:
                        continue
                    t_out.append(abs_ms)
                    bpm_out.append(bpm_arr[i])
                    rr_out.append(rr_arr[i] if i < len(rr_arr) else [])
                    contact_out.append(cont_arr[i] if i < len(cont_arr) else None)
            cur += timedelta(days=1)

        # Day-files are already minute-ordered; samples within a minute are
        # append-ordered. But if the in-memory buffer's minute landed earlier
        # than a later flushed minute on disk (edge case at minute rollover),
        # sort defensively to guarantee monotonic timestamps.
        if any(t_out[i] > t_out[i + 1] for i in range(len(t_out) - 1)):
            order = sorted(range(len(t_out)), key=lambda i: t_out[i])
            t_out = [t_out[i] for i in order]
            bpm_out = [bpm_out[i] for i in order]
            rr_out = [rr_out[i] for i in order]
            contact_out = [contact_out[i] for i in order]

        return {
            "t": t_out,
            "bpm": bpm_out,
            "rr": rr_out,
            "contact": contact_out,
        }

    async def window(
        self, center: datetime, before_sec: float, after_sec: float,
    ) -> dict:
        """Return samples in [center - before_sec, center + after_sec] with
        timestamps as float seconds relative to `center` (negative before).

        Stackable across events: feed many event centers through this and the
        resulting tensors share a common time axis. Used by `GET /hr/around/{id}`
        and the training pipeline.
        """
        start = center - timedelta(seconds=before_sec)
        end = center + timedelta(seconds=after_sec)
        data = await self.read_range(start, end)
        center_ms = int(center.astimezone(timezone.utc).timestamp() * 1000)
        return {
            "t_rel_sec": [(ms - center_ms) / 1000.0 for ms in data["t"]],
            "bpm": data["bpm"],
            "rr": data["rr"],
            "contact": data["contact"],
        }
