"""Real-time heart-rate monitor for a Garmin watch broadcasting over
standard BLE Heart Rate Profile (0x180D).

The watch must have HR broadcasting enabled:
    Settings -> Sensors & Accessories -> Wrist Heart Rate
        -> Broadcast Heart Rate

Run standalone:
    python garmin_hr.py

Use as a library (e.g. from FastAPI):
    from garmin_hr import GarminHRMonitor
    monitor = GarminHRMonitor()           # auto-discovers by name
    await monitor.start()
    print(monitor.latest)                  # last reading or None
    async for reading in monitor.stream(): # live feed
        ...
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from bleak import BleakClient, BleakScanner

HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
BODY_SENSOR_LOCATION_UUID = "00002a38-0000-1000-8000-00805f9b34fb"

logger = logging.getLogger(__name__)


@dataclass
class HRReading:
    bpm: int
    timestamp: datetime
    # None = sensor doesn't report this
    sensor_contact: Optional[bool] = None
    # cumulative kJ since session start
    energy_expended_kj: Optional[int] = None
    # for HRV
    rr_intervals_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bpm": self.bpm,
            "timestamp": self.timestamp.isoformat(),
            "sensor_contact": self.sensor_contact,
            "energy_expended_kj": self.energy_expended_kj,
            "rr_intervals_ms": self.rr_intervals_ms,
        }


def parse_hr_measurement(data: bytes) -> HRReading:
    """Decode a Heart Rate Measurement (0x2A37) notification per BLE spec."""
    flags = data[0]
    uint16_format = bool(flags & 0x01)
    contact_supported = bool(flags & 0x04)
    contact_detected = bool(flags & 0x02)
    energy_present = bool(flags & 0x08)
    rr_present = bool(flags & 0x10)

    offset = 1
    if uint16_format:
        bpm = int.from_bytes(data[offset:offset + 2], "little")
        offset += 2
    else:
        bpm = data[offset]
        offset += 1

    energy = None
    if energy_present:
        energy = int.from_bytes(data[offset:offset + 2], "little")
        offset += 2

    rr_intervals: list[float] = []
    if rr_present:
        while offset + 2 <= len(data):
            raw = int.from_bytes(data[offset:offset + 2], "little")
            rr_intervals.append(raw * 1000.0 / 1024.0)
            offset += 2

    return HRReading(
        bpm=bpm,
        timestamp=datetime.now(timezone.utc),
        sensor_contact=contact_detected if contact_supported else None,
        energy_expended_kj=energy,
        rr_intervals_ms=rr_intervals,
    )


class GarminHRMonitor:
    """Connects to the watch, parses notifications, fans out to subscribers,
    auto-reconnects on disconnect. Lifecycle: start() / stop().
    """

    def __init__(
        self,
        address: Optional[str] = None,
        name_filter: str = "garmin",
        reconnect_delay: float = 5.0,
        scan_timeout: float = 10.0,
        rescan_interval: float = 30.0,
        connect_timeout: float = 20.0,
        clear_bluez_on_retry: bool = True,
        notification_stale_sec: float = 30.0,
        disconnect_timeout: float = 5.0,
    ) -> None:
        self.address = address
        self.name_filter = name_filter.lower()
        self.reconnect_delay = reconnect_delay
        self.scan_timeout = scan_timeout
        # Time to wait between scan attempts when no watch is found.
        # Watch stops broadcasting when not worn, so we keep retrying.
        self.rescan_interval = rescan_interval
        # Bleak's default connect timeout (10s) is short for BlueZ when the
        # watch advertises slowly. Bumped for reliability on Linux.
        self.connect_timeout = connect_timeout
        # Clear stale BlueZ connection state via `bluetoothctl disconnect`
        # before each connect attempt on Linux. Idempotent and best-effort.
        self.clear_bluez_on_retry = clear_bluez_on_retry
        # Watchdog: if the link is up but no HR notifications arrive for this
        # many seconds, force a reconnect. Catches BlueZ phantom-connected
        # state where bleak's disconnect callback never fires.
        self.notification_stale_sec = notification_stale_sec
        # Hard cap on client.disconnect() so a stuck BlueZ teardown can't
        # freeze the reconnect loop. The next iteration's BlueZ-clear will
        # mop up any leaked device entry.
        self.disconnect_timeout = disconnect_timeout

        self._latest: Optional[HRReading] = None
        self._subscribers: list[asyncio.Queue[HRReading]] = []
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._connected = asyncio.Event()
        # monotonic timestamp of last notification (or connect time before any
        # notification arrives). Used by the notification watchdog.
        self._last_notify_at: float = 0.0

    @property
    def latest(self) -> Optional[HRReading]:
        return self._latest

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    async def _resolve_address(self) -> str:
        if self.address:
            return self.address
        logger.info(
            "Scanning %.0fs for device matching %r...",
            self.scan_timeout, self.name_filter,
        )
        devices = await BleakScanner.discover(
            timeout=self.scan_timeout,
            return_adv=True,
        )
        # devices is dict[address, (BLEDevice, AdvertisementData)]
        hr_advertisers: list[tuple[str, str | None]] = []
        name_matches: list[tuple[str, str | None]] = []
        for addr, (dev, adv) in devices.items():
            name = dev.name or adv.local_name
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            advertises_hr = HR_SERVICE_UUID.lower() in uuids
            logger.info(
                "  %s  name=%r  hr_adv=%s",
                addr, name, advertises_hr,
            )
            if advertises_hr:
                hr_advertisers.append((addr, name))
            if name and self.name_filter in name.lower():
                name_matches.append((addr, name))

        if hr_advertisers:
            addr, name = hr_advertisers[0]
            logger.info("Using HR advertiser: %s (%s)", name, addr)
            return addr
        if name_matches:
            addr, name = name_matches[0]
            logger.info(
                "No HR advertisement, falling back to name match: "
                "%s (%s) -- HR may still be reachable if paired",
                name, addr,
            )
            return addr
        raise RuntimeError(
            "No matching device found. On the watch: Settings -> Sensors & "
            "Accessories -> Wrist Heart Rate -> Broadcast Heart Rate -> "
            "Broadcast Now (and leave that screen on). Or pass an address: "
            "python garmin_hr.py <address>"
        )

    def _on_notify(self, _sender: int, data: bytearray) -> None:
        try:
            reading = parse_hr_measurement(bytes(data))
        except Exception:
            logger.exception(
                "Failed to parse HR measurement: %r", bytes(data).hex(),
            )
            return
        self._latest = reading
        self._last_notify_at = time.monotonic()
        for q in list(self._subscribers):
            try:
                q.put_nowait(reading)
            except asyncio.QueueFull:
                # Drop oldest to keep the stream live
                try:
                    q.get_nowait()
                    q.put_nowait(reading)
                except asyncio.QueueEmpty:
                    pass

    async def _bluetoothctl(self, *args: str) -> None:
        """Run `bluetoothctl` with args; log a one-line summary. Best-effort."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "bluetoothctl", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=5.0,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("bluetoothctl %s timed out (>5s)", " ".join(args))
                return
            out = (stdout or b"").decode(errors="replace")
            err = (stderr or b"").decode(errors="replace").strip()
            # `remove` dumps every evicted GATT service; keep only the final summary line.
            tail = next(
                (
                    line.strip()
                    for line in reversed(out.splitlines())
                    if line.strip() and not line.lstrip().startswith("[")
                ),
                out.strip()[:200],
            )
            if proc.returncode == 0 and not err:
                logger.info("bluetoothctl %s: %s", " ".join(args), tail)
            else:
                logger.warning(
                    "bluetoothctl %s rc=%s tail=%r err=%r",
                    " ".join(args), proc.returncode, tail, err,
                )
        except Exception as e:
            logger.warning("bluetoothctl %s failed: %s: %r",
                           " ".join(args), type(e).__name__, e)

    async def _clear_stale_bluez_state(self, address: str) -> None:
        """Force-clear BlueZ state for `address` so the next bleak connect is fresh.

        `disconnect` terminates any active connection; `remove` purges the device
        from BlueZ's registry, which evicts the stale GATT service cache that
        otherwise causes service discovery to time out on reconnect.

        Linux only; silently skipped elsewhere or if `bluetoothctl` is missing.
        Both calls are best-effort — failures are logged but don't block.
        """
        if not self.clear_bluez_on_retry or platform.system() != "Linux":
            return
        if shutil.which("bluetoothctl") is None:
            logger.info("bluetoothctl not on PATH; skipping BlueZ state clear")
            return
        # Order matters: must disconnect before remove if currently connected.
        await self._bluetoothctl("disconnect", address)
        await self._bluetoothctl("remove", address)

    async def _sleep_or_stop(self, seconds: float) -> bool:
        """Sleep up to `seconds`. Returns True if stop was signaled."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
            return True
        except asyncio.TimeoutError:
            return False

    async def _notification_watchdog(self) -> None:
        """Returns when no HR notification has arrived for `notification_stale_sec`.

        Catches the BlueZ phantom-connected case: bleak says we're connected
        and never fires its disconnect callback, but no GATT notifications
        actually flow. Without this, the surrounding `asyncio.wait` would
        block forever and the only fix would be a process restart.
        """
        # Poll often enough to react quickly, but not so often that we burn CPU.
        poll_interval = min(5.0, self.notification_stale_sec / 3)
        while True:
            await asyncio.sleep(poll_interval)
            elapsed = time.monotonic() - self._last_notify_at
            if elapsed > self.notification_stale_sec:
                logger.warning(
                    "No HR notifications for %.1fs (>%ss) — forcing reconnect",
                    elapsed, self.notification_stale_sec,
                )
                return

    async def _run(self) -> None:
        # Auto-discovered addresses are re-scanned on every cycle so a watch
        # that turned its broadcast off (e.g. taken off the wrist) can be
        # picked up again when it resumes. User-supplied addresses are sticky.
        user_supplied = self.address is not None
        while not self._stop.is_set():
            address: Optional[str] = self.address
            if address is None:
                try:
                    address = await self._resolve_address()
                except RuntimeError as e:
                    logger.info(
                        "No watch found; rescanning in %.0fs (%s)",
                        self.rescan_interval, e,
                    )
                    if await self._sleep_or_stop(self.rescan_interval):
                        return
                    continue
                except Exception as e:
                    logger.warning("Scan failed: %s; retrying in %.0fs",
                                   e, self.rescan_interval)
                    if await self._sleep_or_stop(self.rescan_interval):
                        return
                    continue

            # Clear any stale BlueZ connection record from a prior process.
            # Without this, bleak's connect call often times out.
            await self._clear_stale_bluez_state(address)

            disconnected = asyncio.Event()

            def _on_disconnect(_client: BleakClient) -> None:
                disconnected.set()

            try:
                async with BleakClient(
                    address,
                    disconnected_callback=_on_disconnect,
                    timeout=self.connect_timeout,
                ) as client:
                    logger.info("Connected to %s", address)
                    self._connected.set()
                    # Seed the watchdog clock so it doesn't trigger before
                    # the first notification has had a chance to arrive.
                    self._last_notify_at = time.monotonic()
                    await client.start_notify(
                        HR_MEASUREMENT_UUID, self._on_notify,
                    )
                    stop_task = asyncio.create_task(self._stop.wait())
                    disc_task = asyncio.create_task(disconnected.wait())
                    watchdog_task = asyncio.create_task(
                        self._notification_watchdog(),
                    )
                    _, pending = await asyncio.wait(
                        {stop_task, disc_task, watchdog_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    if client.is_connected:
                        try:
                            await client.stop_notify(HR_MEASUREMENT_UUID)
                        except Exception:
                            pass
            except Exception as exc:
                # str(exc) is often empty for bleak/BlueZ errors — log type + repr too.
                logger.warning(
                    "BLE session ended: %s: %r",
                    type(exc).__name__, exc,
                )
            finally:
                self._connected.clear()

            if self._stop.is_set():
                return

            # Sticky user-supplied address retries quickly with the same MAC.
            # Discovered addresses get re-resolved next cycle so we adapt to a
            # watch that re-broadcasts on a different session.
            if user_supplied:
                delay = self.reconnect_delay
                logger.info("Reconnecting in %.0fs...", delay)
            else:
                delay = self.rescan_interval
                logger.info("Rescanning for watch in %.0fs...", delay)
            if await self._sleep_or_stop(delay):
                return

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="garmin-hr-monitor")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(self._task, timeout=10)
        except asyncio.TimeoutError:
            self._task.cancel()
        self._task = None

    async def stream(self, maxsize: int = 100) -> AsyncIterator[HRReading]:
        """Yield readings as they arrive. One queue per consumer."""
        queue: asyncio.Queue[HRReading] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.remove(queue)


async def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    address = sys.argv[1] if len(sys.argv) > 1 else None
    monitor = GarminHRMonitor(address=address)
    await monitor.start()
    try:
        async for r in monitor.stream():
            extras = []
            if r.sensor_contact is not None:
                extras.append(f"contact={r.sensor_contact}")
            if r.rr_intervals_ms:
                rr_str = ",".join(f"{x:.0f}" for x in r.rr_intervals_ms)
                extras.append("rr=" + rr_str)
            tail = (" " + " ".join(extras)) if extras else ""
            print(f"{r.timestamp.isoformat()}  {r.bpm} bpm{tail}")
    finally:
        await monitor.stop()


if __name__ == "__main__":
    try:
        asyncio.run(_cli())
    except KeyboardInterrupt:
        pass
