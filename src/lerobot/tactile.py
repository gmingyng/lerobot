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

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import serial

TACTILE_USB_COUNT = 2
MIN_TACTILE_USB_COUNT = 1
TACTILE_ROUTES_PER_USB = 2
TACTILE_CHANNELS_PER_ROUTE = 4
TACTILE_VALUES_PER_CHANNEL = 8
TACTILE_RAW_VALUE_COUNT = TACTILE_ROUTES_PER_USB * TACTILE_CHANNELS_PER_ROUTE * TACTILE_VALUES_PER_CHANNEL
TACTILE_VALUES_PER_USB = TACTILE_ROUTES_PER_USB * TACTILE_VALUES_PER_CHANNEL
TACTILE_VALUE_COUNT = TACTILE_USB_COUNT * TACTILE_VALUES_PER_USB


def parse_tactile_line(data: bytes | bytearray | str, header: str = "0") -> np.ndarray | None:
    """Parse one USB frame and average each route's four channels column-wise."""
    text = data.strip() if isinstance(data, str) else bytes(data).decode("utf-8", errors="ignore").strip()
    if not text:
        return None

    parts = [part.strip() for part in text.split(",")]
    if len(parts) < TACTILE_RAW_VALUE_COUNT + 1 or parts[0] != header:
        return None

    try:
        values = np.asarray(parts[1 : TACTILE_RAW_VALUE_COUNT + 1], dtype=np.float32)
    except ValueError:
        return None

    routes = values.reshape(
        TACTILE_ROUTES_PER_USB,
        TACTILE_CHANNELS_PER_ROUTE,
        TACTILE_VALUES_PER_CHANNEL,
    )
    return routes.mean(axis=1, dtype=np.float32).reshape(TACTILE_VALUES_PER_USB)


def tactile_feature(usb_count: int = TACTILE_USB_COUNT) -> dict[str, dict]:
    if not MIN_TACTILE_USB_COUNT <= usb_count <= TACTILE_USB_COUNT:
        raise ValueError(
            f"Tactile USB count must be between {MIN_TACTILE_USB_COUNT} and {TACTILE_USB_COUNT}, "
            f"got {usb_count}."
        )

    names = [
        f"usb{usb_index}_{route_name}_{value_index}"
        for usb_index in range(1, usb_count + 1)
        for route_name in ("f1", "f2")
        for value_index in range(1, TACTILE_VALUES_PER_CHANNEL + 1)
    ]
    return {
        "tactile": {
            "dtype": "float32",
            "shape": (usb_count * TACTILE_VALUES_PER_USB,),
            "names": names,
        }
    }


@dataclass
class TactileConfig:
    ports: list[str] = field(default_factory=lambda: ["/dev/ttyCH341USB0", "/dev/ttyCH341USB1"])
    baudrate: int = 115200
    serial_timeout_s: float = 0.2
    startup_timeout_s: float = 5.0
    stale_timeout_s: float = 1.0
    header: str = "0"

    def __post_init__(self):
        if not MIN_TACTILE_USB_COUNT <= len(self.ports) <= TACTILE_USB_COUNT:
            raise ValueError(f"Expected one or two tactile USB ports, got {self.ports}.")
        if len(set(self.ports)) != len(self.ports):
            raise ValueError(f"Tactile USB ports must be unique, got {self.ports}.")
        if self.serial_timeout_s <= 0 or self.startup_timeout_s <= 0 or self.stale_timeout_s <= 0:
            raise ValueError("Tactile serial, startup, and stale timeouts must all be positive.")


class TactileReader:
    """Continuously read one or two tactile USB adapters and expose their latest averaged values."""

    def __init__(self, config: TactileConfig):
        self.config = config
        self.usb_count = len(config.ports)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._serial_ports: list[serial.Serial] = []
        self._threads: list[threading.Thread] = []
        self._latest_values: list[np.ndarray | None] = [None] * self.usb_count
        self._updated_at: list[float | None] = [None] * self.usb_count
        self._errors: list[str | None] = [None] * self.usb_count

    def connect(self) -> None:
        logging.info("Opening tactile USB ports %s at %s baud", self.config.ports, self.config.baudrate)
        try:
            for port in self.config.ports:
                serial_port = serial.Serial(
                    port=port,
                    baudrate=self.config.baudrate,
                    timeout=self.config.serial_timeout_s,
                )
                serial_port.reset_input_buffer()
                self._serial_ports.append(serial_port)

            for usb_index, serial_port in enumerate(self._serial_ports):
                thread = threading.Thread(
                    target=self._read_loop,
                    args=(usb_index, serial_port),
                    daemon=True,
                    name=f"tactile-{usb_index + 1}",
                )
                thread.start()
                self._threads.append(thread)

            deadline = time.monotonic() + self.config.startup_timeout_s
            while time.monotonic() < deadline:
                with self._lock:
                    errors = self._active_errors()
                    ready = all(values is not None for values in self._latest_values)
                if errors:
                    raise RuntimeError(f"Tactile reader failed during startup: {errors}")
                if ready:
                    logging.info("Received initial tactile frames from %s USB port(s).", self.usb_count)
                    return
                time.sleep(0.01)

            missing_ports = [
                port
                for port, values in zip(self.config.ports, self._latest_values, strict=True)
                if values is None
            ]
            raise TimeoutError(
                f"No valid tactile frame received from {missing_ports} within {self.config.startup_timeout_s}s."
            )
        except Exception:
            self.disconnect()
            raise

    def _read_loop(self, usb_index: int, serial_port: serial.Serial) -> None:
        try:
            while not self._stop_event.is_set():
                raw_line = serial_port.readline()
                if not raw_line:
                    continue

                values = parse_tactile_line(raw_line, self.config.header)
                if values is None:
                    continue

                with self._lock:
                    self._latest_values[usb_index] = values
                    self._updated_at[usb_index] = time.monotonic()
                    self._errors[usb_index] = None
        except Exception as exc:
            if not self._stop_event.is_set():
                with self._lock:
                    self._errors[usb_index] = f"{self.config.ports[usb_index]}: {exc}"

    def read(self) -> np.ndarray:
        with self._lock:
            errors = self._active_errors()
            if errors:
                raise RuntimeError(f"Tactile reader stopped: {errors}")

            if any(values is None for values in self._latest_values):
                raise RuntimeError("Tactile values requested before every USB port produced a valid frame.")

            now = time.monotonic()
            stale_ports = [
                port
                for port, updated_at in zip(self.config.ports, self._updated_at, strict=True)
                if updated_at is None or now - updated_at > self.config.stale_timeout_s
            ]
            if stale_ports:
                raise RuntimeError(
                    f"Tactile data from {stale_ports} is older than {self.config.stale_timeout_s}s."
                )

            values = [value for value in self._latest_values if value is not None]
            return np.concatenate(values).astype(np.float32, copy=False)

    def _active_errors(self) -> list[str]:
        return [error for error in self._errors if error is not None]

    def disconnect(self) -> None:
        self._stop_event.set()
        for serial_port in self._serial_ports:
            if serial_port.is_open:
                serial_port.close()
        for thread in self._threads:
            thread.join(timeout=self.config.serial_timeout_s + 0.2)
        self._threads.clear()
        self._serial_ports.clear()
