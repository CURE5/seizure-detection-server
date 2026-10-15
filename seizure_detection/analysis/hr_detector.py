"""HR-based seizure detector.

Consumes a stream of `HRReading` from `GarminHRMonitor`, maintains a rolling
trimmed-median baseline, and fires an alert when BPM stays above either:

  * baseline + DELTA_THRESHOLD, or
  * ABSOLUTE_THRESHOLD,

continuously for at least SUSTAIN_SEC.

Baseline updates are paused while readings are above either threshold, so a
real event cannot drag the baseline upward and silently de-trigger.

Tuning is patient-specific: defaults assume resting ~77 bpm with low natural
variability.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Optional

from ..config import GarminHRConfig
from ..garmin_hr import GarminHRMonitor, HRReading
from ..logging import HRLogger

logger = logging.getLogger(__name__)


AlertCallback = Callable[[dict], Awaitable[None]]


@dataclass
class HRState:
    bpm: Optional[int]
    baseline_bpm: Optional[float]
    above_baseline: bool
    above_absolute: bool
    sustained_sec: float
    last_alert_at: Optional[datetime]
    recent: list[tuple[datetime, int]]


class HRDetector:
    """Long-lived consumer of `monitor.stream()`. Owns baseline + sustain logic."""

    def __init__(
        self,
        monitor: GarminHRMonitor,
        on_alert: Optional[AlertCallback] = None,
        baseline_window_sec: int = GarminHRConfig.BASELINE_WINDOW_SEC,
        delta_threshold: int = GarminHRConfig.DELTA_THRESHOLD,
        absolute_threshold: int = GarminHRConfig.ABSOLUTE_THRESHOLD,
        sustain_sec: int = GarminHRConfig.SUSTAIN_SEC,
        alert_cooldown_sec: int = GarminHRConfig.ALERT_COOLDOWN_SEC,
        recent_history_sec: int = 120,
        hr_logger: Optional[HRLogger] = None,
    ) -> None:
        self.monitor = monitor
        self.on_alert = on_alert
        self.baseline_window = timedelta(seconds=baseline_window_sec)
        self.delta_threshold = delta_threshold
        self.absolute_threshold = absolute_threshold
        self.sustain_sec = sustain_sec
        self.alert_cooldown = timedelta(seconds=alert_cooldown_sec)
        self.recent_history = timedelta(seconds=recent_history_sec)
        self.hr_logger = hr_logger or HRLogger()

        # (timestamp, bpm) — only contains readings taken while at rest.
        self._baseline_buf: deque[tuple[datetime, int]] = deque()
        # (timestamp, bpm) — short window for overlay rendering.
        self._recent_buf: deque[tuple[datetime, int]] = deque()

        self._above_since: Optional[datetime] = None
        self._sustained_sec: float = 0.0
        self._last_alert_at: Optional[datetime] = None
        self._alert_active: bool = False  # de-duplicates within one elevation

    @property
    def baseline_bpm(self) -> Optional[float]:
        n = len(self._baseline_buf)
        if n == 0:
            return None
        sorted_vals = sorted(b for _, b in self._baseline_buf)
        if n < 10:
            return float(sorted_vals[n // 2])
        trim = n // 10
        trimmed = sorted_vals[trim:n - trim]
        return float(trimmed[len(trimmed) // 2])

    def _is_above(self, bpm: int, baseline: Optional[float]) -> tuple[bool, bool]:
        above_abs = bpm >= self.absolute_threshold
        above_delta = baseline is not None and bpm >= baseline + self.delta_threshold
        return above_delta, above_abs

    def state(self) -> HRState:
        bpm: Optional[int] = None
        if self.monitor.latest is not None:
            bpm = self.monitor.latest.bpm
        baseline = self.baseline_bpm
        above_d, above_a = (False, False)
        if bpm is not None:
            above_d, above_a = self._is_above(bpm, baseline)
        return HRState(
            bpm=bpm,
            baseline_bpm=baseline,
            above_baseline=above_d,
            above_absolute=above_a,
            sustained_sec=self._sustained_sec,
            last_alert_at=self._last_alert_at,
            recent=list(self._recent_buf),
        )

    def _prune(self, buf: deque[tuple[datetime, int]], window: timedelta, now: datetime) -> None:
        cutoff = now - window
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    async def _process(self, reading: HRReading) -> None:
        now = reading.timestamp
        bpm = reading.bpm

        # Persist the raw reading first. Derived fields (baseline, above_*) are
        # recomputed at read time, so we only store the sensor sample itself.
        # Log even when contact is False so the gap is visible in post-hoc review.
        try:
            await self.hr_logger.add(reading)
        except Exception:
            logger.exception("HR logger.add failed")

        # Sensor reports no skin contact: BPM is the strap's last-known value,
        # not actual heart rate. Don't pollute the baseline and don't let it
        # trip the elevation gate. Reset any in-flight elevation so we don't
        # carry phantom sustained_sec across a contact-loss gap.
        if reading.sensor_contact is False:
            self._above_since = None
            self._sustained_sec = 0.0
            self._alert_active = False
            return

        self._recent_buf.append((now, bpm))
        self._prune(self._recent_buf, self.recent_history, now)

        baseline = self.baseline_bpm
        above_d, above_a = self._is_above(bpm, baseline)
        is_above = above_d or above_a

        if is_above:
            if self._above_since is None:
                self._above_since = now
            self._sustained_sec = (now - self._above_since).total_seconds()
        else:
            self._above_since = None
            self._sustained_sec = 0.0
            self._alert_active = False
            # Only update baseline while at rest.
            self._baseline_buf.append((now, bpm))
            self._prune(self._baseline_buf, self.baseline_window, now)

        if (
            is_above
            and self._sustained_sec >= self.sustain_sec
            and not self._alert_active
            and (
                self._last_alert_at is None
                or (now - self._last_alert_at) >= self.alert_cooldown
            )
        ):
            await self._fire_alert(reading, baseline, above_d, above_a)

    async def _fire_alert(
        self,
        reading: HRReading,
        baseline: Optional[float],
        above_delta: bool,
        above_absolute: bool,
    ) -> None:
        self._last_alert_at = reading.timestamp
        self._alert_active = True

        delta = (
            (reading.bpm - baseline) if baseline is not None else None
        )
        trigger = "delta" if above_delta else "absolute"
        alert: dict = {
            "type": "hr_alert",
            "alert_type": "onset",
            "timestamp": reading.timestamp.isoformat(),
            "confidence": 1.0,
            "stage": "hr_sudden",
            "time_weight": 1.0,
            "bpm": reading.bpm,
            "baseline_bpm": baseline,
            "delta": delta,
            "trigger": trigger,
            "sustained_sec": self._sustained_sec,
        }
        logger.warning(
            "HR alert: bpm=%d baseline=%s delta=%s sustained=%.1fs trigger=%s",
            reading.bpm,
            f"{baseline:.1f}" if baseline is not None else "n/a",
            f"{delta:.1f}" if delta is not None else "n/a",
            self._sustained_sec,
            trigger,
        )
        if self.on_alert is not None:
            try:
                await self.on_alert(alert)
            except Exception as e:
                logger.exception("HR alert callback failed: %s", e)
        # Get the partial-minute buffer onto disk immediately so the event is
        # available for review even if the process dies.
        await self.hr_logger.flush()

    async def run(self) -> None:
        """Consume monitor.stream() until cancelled."""
        logger.info(
            "HR detector running: baseline_window=%ds delta=+%d abs=>=%d sustain=%ds",
            int(self.baseline_window.total_seconds()),
            self.delta_threshold,
            self.absolute_threshold,
            self.sustain_sec,
        )
        await self.hr_logger.start()
        try:
            async for reading in self.monitor.stream():
                try:
                    await self._process(reading)
                except Exception:
                    logger.exception("HR detector failed to process reading")
        except asyncio.CancelledError:
            logger.info("HR detector cancelled")
            raise
        finally:
            await self.hr_logger.stop()
