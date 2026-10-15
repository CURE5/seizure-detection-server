"""Standalone HR probe: scan, connect, subscribe, and dump notifications.

A/B test against the Pi setup. Run from any host with `bleak` available
(e.g. the Mac with the watch right in front of you):

    pip install bleak
    python scripts/hr_probe.py

Note: macOS doesn't expose MAC addresses — bleak resolves devices via opaque
CoreBluetooth UUIDs. We discover by name match instead, the same fallback the
production code uses when no MAC is configured.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone

from bleak import BleakClient, BleakScanner

HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
NAME_FRAGMENTS = ("garmin", "vivoactive", "vívoactive", "vívoactiv", "vivoactiv")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def parse_hr(data: bytes) -> tuple[int, bool | None]:
    flags = data[0]
    uint16 = bool(flags & 0x01)
    contact_supported = bool(flags & 0x04)
    contact = bool(flags & 0x02) if contact_supported else None
    bpm = (
        int.from_bytes(data[1:3], "little") if uint16 else data[1]
    )
    return bpm, contact


async def main() -> int:
    print(f"[{now()}] Scanning 10s for any HR-broadcasting device…")
    devices = await BleakScanner.discover(timeout=10.0, return_adv=True)

    candidates = []
    for addr, (dev, adv) in devices.items():
        name = dev.name or adv.local_name or ""
        uuids = [u.lower() for u in (adv.service_uuids or [])]
        is_hr = HR_SERVICE_UUID.lower() in uuids
        name_match = any(frag in name.lower() for frag in NAME_FRAGMENTS)
        if is_hr or name_match:
            candidates.append((dev, adv, is_hr, name_match))
            print(
                f"[{now()}]   addr={addr}  name={name!r}  "
                f"rssi={adv.rssi}  hr_adv={is_hr}  name_match={name_match}"
            )

    if not candidates:
        print(f"[{now()}] No matching device. Make sure Broadcast HR screen is on.")
        return 1

    # Prefer HR-advertising devices over name-only matches.
    candidates.sort(key=lambda t: (not t[2], not t[3]))
    target_dev = candidates[0][0]
    print(f"[{now()}] Connecting to {target_dev.name!r} ({target_dev.address})…")

    t0 = time.monotonic()
    last_notify = 0.0
    notify_count = 0

    def on_hr(_sender, data: bytearray) -> None:
        nonlocal last_notify, notify_count
        last_notify = time.monotonic()
        notify_count += 1
        bpm, contact = parse_hr(bytes(data))
        print(
            f"[{now()}] HR #{notify_count:3d}  bpm={bpm}  contact={contact}  "
            f"raw={bytes(data).hex()}"
        )

    try:
        async with BleakClient(target_dev, timeout=20.0) as client:
            t_connected = time.monotonic()
            print(
                f"[{now()}] Connected in {t_connected - t0:.2f}s  "
                f"(is_connected={client.is_connected})"
            )
            await client.start_notify(HR_MEASUREMENT_UUID, on_hr)
            print(f"[{now()}] Subscribed to HR notifications. Streaming for 30s…")
            await asyncio.sleep(30)
            try:
                await client.stop_notify(HR_MEASUREMENT_UUID)
            except Exception:
                pass
        elapsed = time.monotonic() - t_connected
        print(
            f"[{now()}] Session ended cleanly. "
            f"duration={elapsed:.1f}s  notifications={notify_count}"
        )
        return 0 if notify_count > 0 else 2
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(
            f"[{now()}] FAILURE after {elapsed:.2f}s  "
            f"{type(exc).__name__}: {exc!r}"
        )
        return 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
