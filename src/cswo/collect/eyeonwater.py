"""EyeOnWater (NIOZ/MARIS): full JSON dump from the open API, no authentication.

Holds Forel-Ule index, optional Secchi depth and a photo per record; there is no free-text field.
Run: uv run python -m cswo.collect.eyeonwater
"""
from __future__ import annotations

import json
from datetime import date

from cswo import RAW
from cswo.collect.common import session
from cswo.licence import log_download

URL = "https://www.eyeonwater.org/api/observations?period=0&format=json"   # period=0 -> complete dataset
LICENCE_NOTE = "EyeOnWater open data policy (https://www.eyeonwater.org/observations/downloads); re-check at download"


def main() -> None:
    out = RAW / "eyeonwater" / f"eyeonwater_full_{date.today():%Y%m%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = session().get(URL, timeout=600)
    r.raise_for_status()
    data = r.json()
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    log_download("eyeonwater", URL, "EyeOnWater-open", len(data), notes=LICENCE_NOTE)
    print(f"EyeOnWater: {len(data)} records -> {out}")


if __name__ == "__main__":
    main()
