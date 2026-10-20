"""Waiting for a starting ninferno-serve: /health answers {"status": "ok"} once the model is loaded."""

from __future__ import annotations

import http.client
import json
import subprocess
import time
from typing import Any, Callable
import urllib.parse


class ServeNotReady(RuntimeError):
    """serve exited during startup, or was not healthy within the time given."""


def health(base_url: str, timeout: float = 2.0) -> dict[str, Any] | None:
    """The /health body when it says ok; None while serve does not answer, or answers otherwise."""
    parsed = urllib.parse.urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 80,
                                            timeout=timeout)
    try:
        connection.request("GET", "/health", headers={"Connection": "close"})
        response = connection.getresponse()
        body = response.read()
    except (OSError, http.client.HTTPException):
        return None
    finally:
        connection.close()
    if response.status != 200:
        return None
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("status") == "ok" else None


def wait_healthy(base_url: str, timeout: float, process: subprocess.Popen | None = None,
                 waiting: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Polls /health until it says ok and returns the body (it carries the load gauges). Raises
    ServeNotReady when `process` exits first or `timeout` seconds pass; `waiting` is called with the
    seconds elapsed between polls."""
    started = time.monotonic()
    while True:
        if process is not None and process.poll() is not None:
            raise ServeNotReady(f"ninferno-serve exited during startup with status {process.returncode}")
        value = health(base_url)
        if value is not None:
            return value
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            raise ServeNotReady(f"ninferno-serve at {base_url} was not healthy after {timeout:g}s")
        if waiting is not None:
            waiting(elapsed)
        time.sleep(0.1)
