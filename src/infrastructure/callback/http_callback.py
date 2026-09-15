from __future__ import annotations

import json
import time
import urllib.request


class HttpCallbackAdapter:
    def __init__(self, timeout_s: float = 5.0, max_retries: int = 3, backoff_s: float = 0.25) -> None:
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_s = backoff_s

    def send(self, callback_url: str, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        last_exc = None
        for i in range(self._max_retries):
            try:
                req = urllib.request.Request(
                    callback_url,
                    data=body,
                    headers={"content-type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout_s):
                    return
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                if i + 1 < self._max_retries:
                    time.sleep(self._backoff_s * (2**i))
        if last_exc:
            raise last_exc
