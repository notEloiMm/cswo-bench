"""Licence normalisation, redistribution whitelist and the per-download licence log."""
from __future__ import annotations

import csv
import re
from datetime import datetime, timezone

from cswo import DATA

LOG_PATH = DATA / "licence_log.csv"
LOG_FIELDS = ["timestamp_utc", "platform", "source_url", "licence", "doi", "n_records", "notes"]

# Licences that allow redistribution of the record text in a public benchmark.
# CC-BY-NC is excluded by default (spec 1 says "permits redistribution"; NC restricts reuse of the benchmark).
REDISTRIBUTABLE = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "ODC-By-1.0", "ODbL-1.0", "EyeOnWater-open"}
NON_COMMERCIAL = {"CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-NC-ND-4.0"}
# Share-alike: a released database containing these records must carry the same licence (data card must say so).
SHARE_ALIKE = {"CC-BY-SA-4.0", "ODbL-1.0"}

_PATTERNS = [
    (r"odbl|open database licen[cs]e", "ODbL-1.0"),
    (r"cc0|publicdomain/zero|public domain", "CC0-1.0"),
    (r"by[-_ ]?nc[-_ ]?nd", "CC-BY-NC-ND-4.0"),
    (r"by[-_ ]?nc[-_ ]?sa", "CC-BY-NC-SA-4.0"),
    (r"by[-_ ]?nc", "CC-BY-NC-4.0"),
    (r"by[-_ ]?nd", "CC-BY-ND-4.0"),
    (r"by[-_ ]?sa", "CC-BY-SA-4.0"),
    (r"odc[-_ ]?by", "ODC-By-1.0"),
    (r"cc[-_ ]?by|licenses/by/", "CC-BY-4.0"),
]


def normalise(raw: str | None) -> str | None:
    """Map platform licence strings (URLs, 'cc-by-nc', 'CC_BY_4_0', ...) to one code."""
    if not raw:
        return None
    s = raw.strip().lower()
    for pat, code in _PATTERNS:
        if re.search(pat, s):
            return code
    return None


def redistributable(code: str | None, allow_nc: bool = False) -> bool:
    if code is None:
        return False
    return code in REDISTRIBUTABLE or (allow_nc and code in NON_COMMERCIAL)


def log_download(platform: str, source_url: str, licence: str, n_records: int,
                 doi: str = "", notes: str = "") -> None:
    """Append one row per download. Start this log with the very first download (course note)."""
    new = not LOG_PATH.exists()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": platform, "source_url": source_url, "licence": licence,
            "doi": doi, "n_records": n_records, "notes": notes,
        })
