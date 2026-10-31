"""Shared helpers for collectors: polite HTTP session, JSONL I/O, config loading."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Iterator

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cswo import CONFIGS

USER_AGENT = "CSWO-Bench/0.1 (University of Oulu NLP course project; research use)"


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    retry = Retry(total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval, self._last = min_interval, 0.0

    def wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()


def config(name: str):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
