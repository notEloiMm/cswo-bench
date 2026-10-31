"""GBIF SQL download for the Nordic/Baltic and European citizen-science portals (spec 1).

GBIF SQL downloads filter ``occurrenceremarks IS NOT NULL`` server side, so the result holds only records with
notes (a full Darwin Core Archive of these taxa and datasets would be ~100M rows, since GBIF predicate downloads
cannot filter on occurrenceRemarks). Each download is rerunnable and mints a DOI, written to the licence log.

Taxa are filtered by rank *names* (configs/aquatic_taxa.yaml, ``gbif_xr``): GBIF occurrences are interpreted
against the Catalogue of Life XR checklist, whose keys differ from the old backbone integer keys.

Credentials come from .env (GBIF_USER, GBIF_PWD, GBIF_EMAIL); see .env.example.
Run: uv run python -m cswo.collect.gbif submit   then   uv run python -m cswo.collect.gbif fetch <download key>
"""
from __future__ import annotations

import argparse
import json
import os
import time

from dotenv import load_dotenv

from cswo import RAW
from cswo.collect.common import config, session
from cswo.licence import log_download

load_dotenv()
API = "https://api.gbif.org/v1/occurrence/download"
LICENCES = ["CC0_1_0", "CC_BY_4_0"]
BIRDS = {"Anatidae", "Gaviidae", "Podicipedidae", "Laridae", "Ardeidae", "Alcedinidae", "Rallidae", "Cinclidae"}
SQL_COLUMNS = [
    "gbifid", "datasetkey", "occurrenceid", "countrycode", "decimallatitude", "decimallongitude", "eventdate",
    "occurrenceremarks", "eventremarks", "fieldnotes", "habitat", "locality", "waterbody",
    "species", "scientificname", "phylum", '"class"', '"order"', "family", "subfamily", "genus",
    "license", "recordedby", "mediatype", "v_associatedmedia", "basisofrecord",
]   # year/month/references are reserved words and are left out (eventdate covers the date)


def _auth() -> tuple[str, str]:
    user, pwd = os.environ.get("GBIF_USER"), os.environ.get("GBIF_PWD")
    if not user or not pwd:
        raise SystemExit("Set GBIF_USER and GBIF_PWD in .env (GBIF username, not e-mail).")
    return user, pwd


def _taxa_by_rank(no_birds: bool) -> dict[str, list[str]]:
    by_rank: dict[str, list[str]] = {}
    for t in config("aquatic_taxa"):
        if no_birds and t["name"] in BIRDS:
            continue
        for x in t["gbif_xr"]:
            by_rank.setdefault(x["rank"], []).append(x["name"])
    return by_rank


def build_sql(no_birds: bool, since: int | None) -> str:
    datasets = ", ".join(f"'{d['key']}'" for d in config("gbif_datasets"))
    licences = ", ".join(f"'{lic}'" for lic in LICENCES)
    quote = lambda col: f'"{col}"' if col in {"class", "order"} else col
    taxa = " OR ".join(f"{quote(rank)} IN ({', '.join(repr(n) for n in names)})"
                       for rank, names in _taxa_by_rank(no_birds).items())
    where = [f"datasetkey IN ({datasets})", f"license IN ({licences})", f"({taxa})",
             "occurrenceremarks IS NOT NULL", "occurrenceremarks <> ''"]
    if since:
        where.append(f'"year" >= {since}')
    return f"SELECT {', '.join(SQL_COLUMNS)} FROM occurrence WHERE " + " AND ".join(where)


def submit(no_birds: bool, since: int | None) -> str:
    body = {"sendNotification": True, "notificationAddresses": [os.environ.get("GBIF_EMAIL", "")],
            "format": "SQL_TSV_ZIP", "sql": build_sql(no_birds, since)}
    r = session().post(f"{API}/request", json=body, auth=_auth(), timeout=120)
    if r.status_code != 201:
        raise SystemExit(f"GBIF refused the request ({r.status_code}): {r.text[:500]}")
    key = r.text.strip()
    out = RAW / "gbif" / f"request_{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"Submitted download {key}. Fetch with: uv run python -m cswo.collect.gbif fetch {key}")
    return key


def fetch(key: str, poll: int = 60) -> None:
    s = session()
    while True:
        meta = s.get(f"{API}/{key}", timeout=60).json()
        status = meta["status"]
        print(f"{key}: {status}")
        if status == "SUCCEEDED":
            break
        if status in {"FAILED", "KILLED", "CANCELLED", "FILE_ERASED"}:
            raise SystemExit(f"Download {key} ended with status {status}")
        time.sleep(poll)
    out = RAW / "gbif" / f"{key}.zip"
    with s.get(meta["downloadLink"], stream=True, timeout=600) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    (RAW / "gbif" / f"{key}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log_download("gbif", f"https://www.gbif.org/occurrence/download/{key}", "CC0-1.0|CC-BY-4.0 (per record)",
                 meta.get("totalRecords", 0), doi=meta.get("doi", ""),
                 notes=f"format={meta.get('request', {}).get('format')}; datasets from configs/gbif_datasets.yaml")
    print(f"Saved {out} (DOI {meta.get('doi')})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("submit")
    a.add_argument("--no-birds", action="store_true", help="drop waterbird families (they dominate counts)")
    a.add_argument("--since", type=int, help="earliest year")
    sub.add_parser("show-sql").add_argument("--no-birds", action="store_true")
    f = sub.add_parser("fetch")
    f.add_argument("key")
    args = ap.parse_args()
    if args.cmd == "submit":
        submit(args.no_birds, args.since)
    elif args.cmd == "show-sql":
        print(build_sql(args.no_birds, None))
    else:
        fetch(args.key)


if __name__ == "__main__":
    main()
