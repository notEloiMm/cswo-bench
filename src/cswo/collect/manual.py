"""Register a hand-downloaded source (CrowdWater, Water Rangers) in the licence log.

Put the exported CSV(s) in data/raw/<platform>/, set the licence in configs/manual_sources.yaml, then run:
uv run python -m cswo.collect.manual crowdwater
"""
import csv
import sys

from cswo import RAW
from cswo.collect.common import config
from cswo.licence import log_download


def main() -> None:
    platform = sys.argv[1]
    meta = config("manual_sources")[platform]
    if not meta.get("licence"):
        raise SystemExit(f"Set the {platform} licence in configs/manual_sources.yaml first.")
    files = sorted((RAW / platform).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV in {RAW / platform}")
    for p in files:
        with p.open(encoding="utf-8-sig") as f:
            n = sum(1 for _ in csv.reader(f)) - 1
        print(f"{p.name}: {n} rows; columns: {open(p, encoding='utf-8-sig').readline().strip()[:300]}")
        log_download(platform, meta.get("licence_url") or "manual download", meta["licence"], n,
                     notes=f"file={p.name}; downloaded_by={meta.get('downloaded_by')}")


if __name__ == "__main__":
    main()
