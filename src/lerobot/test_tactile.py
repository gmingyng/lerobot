#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Display one or two USB tactile sensors using the same processing as record_tac."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from serial.tools import list_ports

from lerobot.tactile import (
    MIN_TACTILE_USB_COUNT,
    TACTILE_ROUTES_PER_USB,
    TACTILE_USB_COUNT,
    TACTILE_VALUES_PER_CHANNEL,
    TactileConfig,
    TactileReader,
)

CH340_VID = 0x1A86


def _is_ch340_port(port_info: Any) -> bool:
    text = " ".join(
        str(value)
        for value in (
            getattr(port_info, "description", ""),
            getattr(port_info, "manufacturer", ""),
            getattr(port_info, "product", ""),
            getattr(port_info, "hwid", ""),
        )
        if value
    ).upper()
    return getattr(port_info, "vid", None) == CH340_VID or any(
        marker in text for marker in ("CH340", "CH341", "QINHENG")
    )


def _format_port(port_info: Any) -> str:
    description = getattr(port_info, "description", "") or getattr(port_info, "hwid", "")
    return f"{port_info.device}: {description}"


def list_available_ports() -> list[Any]:
    return list(list_ports.comports())


def choose_ports(requested_ports: list[str] | None) -> list[str]:
    if requested_ports:
        if not MIN_TACTILE_USB_COUNT <= len(requested_ports) <= TACTILE_USB_COUNT:
            raise SystemExit(f"Pass --port once or twice, got {requested_ports}.")
        if len(set(requested_ports)) != len(requested_ports):
            raise SystemExit(f"Tactile USB ports must be unique, got {requested_ports}.")
        return requested_ports

    configured_ports = TactileConfig().ports
    if all(Path(port).exists() for port in configured_ports):
        return configured_ports

    available_ports = list_available_ports()
    ch340_ports = [str(port.device) for port in available_ports if _is_ch340_port(port)]
    if MIN_TACTILE_USB_COUNT <= len(ch340_ports) <= TACTILE_USB_COUNT:
        return ch340_ports

    if MIN_TACTILE_USB_COUNT <= len(available_ports) <= TACTILE_USB_COUNT:
        return [str(port.device) for port in available_ports]

    available_text = "\n".join(f"  - {_format_port(port)}" for port in available_ports)
    if not available_text:
        available_text = "  (none)"
    raise SystemExit(
        "Could not uniquely select one or two tactile USB ports. Pass --port once or twice.\n"
        f"Available serial ports:\n{available_text}"
    )


def _format_values(values: np.ndarray) -> str:
    return " ".join(f"{value:9.2f}" for value in values)


def display_tactile(values: np.ndarray, ports: list[str], clear: bool) -> None:
    tactile = values.reshape(len(ports), TACTILE_ROUTES_PER_USB, TACTILE_VALUES_PER_CHANNEL)
    if clear and sys.stdout.isatty():
        print("\033[2J\033[H", end="")

    print(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Each F1/F2 value is the column-wise average of four tactile channels.\n")
    for usb_index, port in enumerate(ports):
        print(f"USB{usb_index + 1}: {port}")
        print(f"  F1 avg: {_format_values(tactile[usb_index, 0])}")
        print(f"  F2 avg: {_format_values(tactile[usb_index, 1])}")
        print()

    print(f"Dataset tactile[{values.size}]:")
    print(f"  {_format_values(values)}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read and display one or two USB tactile sensors.")
    parser.add_argument(
        "--port",
        action="append",
        help="Tactile serial port. Pass once or twice. When omitted, CH340/CH341 ports are detected.",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate.")
    parser.add_argument("--header", default="0", help="Accepted tactile frame header.")
    parser.add_argument("--refresh-hz", type=float, default=10.0, help="Display refresh rate.")
    parser.add_argument("--serial-timeout-s", type=float, default=0.2, help="Serial read timeout.")
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=5.0,
        help="Maximum time to wait for the first valid frame from every USB port.",
    )
    parser.add_argument(
        "--stale-timeout-s",
        type=float,
        default=1.0,
        help="Fail when a USB port has not produced a valid frame within this time.",
    )
    parser.add_argument("--once", action="store_true", help="Display one sample and exit.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between samples.")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.list_ports:
        ports = list_available_ports()
        print("\n".join(_format_port(port) for port in ports) if ports else "No serial ports found.")
        return 0

    if args.refresh_hz <= 0:
        raise SystemExit("--refresh-hz must be positive.")

    ports = choose_ports(args.port)
    config = TactileConfig(
        ports=ports,
        baudrate=args.baudrate,
        serial_timeout_s=args.serial_timeout_s,
        startup_timeout_s=args.startup_timeout_s,
        stale_timeout_s=args.stale_timeout_s,
        header=args.header,
    )
    reader = TactileReader(config)

    try:
        reader.connect()
        while True:
            started_at = time.monotonic()
            display_tactile(reader.read(), ports, clear=not args.no_clear)
            if args.once:
                break
            time.sleep(max(0.0, 1.0 / args.refresh_hz - (time.monotonic() - started_at)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
