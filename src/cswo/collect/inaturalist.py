"""iNaturalist API: aquatic-taxa observations with CC0/CC-BY licence, per country, with observer descriptions.

The API has no "has description" filter, so we page through everything (id_above cursor, 200/page,
<= 1 request/s as iNat asks) and keep a slim copy of each record. Description filtering happens at harmonization,
so the raw file also feeds the image-only share in the spec 2 table. Resumable: reruns continue from the last id.
Run: uv run python -m cswo.collect.inaturalist [--countries FI SE ...]
"""
from __future__ import annotations

import argparse

from tqdm import tqdm

from cswo import RAW
from cswo.collect.common import RateLimiter, config, read_jsonl, session, write_jsonl
from cswo.licence import log_download

API = "https://api.inaturalist.org/v1/observations"
LICENCES = "cc0,cc-by"


def slim(o: dict) -> dict:
    """Keep only what the schema needs; drop names/logins except the numeric user id (hashed later)."""
    return {
        "id": o["id"], "uri": o.get("uri"), "license_code": o.get("license_code"),
        "description": o.get("description"), "observed_on": o.get("observed_on"),
        "time_observed_at": o.get("time_observed_at"), "location": o.get("location"),
        "obscured": o.get("obscured"), "geoprivacy": o.get("geoprivacy"),
        "place_guess": o.get("place_guess"), "quality_grade": o.get("quality_grade"),
        "taxon": {k: (o.get("taxon") or {}).get(k) for k in ("id", "name", "rank", "ancestor_ids", "iconic_taxon_name")},
        "user_id": (o.get("user") or {}).get("id"),
        "photos": [{"id": p["id"], "url": p.get("url"), "license_code": p.get("license_code")}
                   for p in o.get("photos", [])],
    }


def collect_country(cc: str, place_id: int, cap: int, taxon_ids: str) -> None:
    out = RAW / "inaturalist" / f"inat_{cc}.jsonl"
    last_id, n = 0, 0
    if out.exists():
        for r in read_jsonl(out):
            last_id, n = max(last_id, r["id"]), n + 1
    s, rl = session(), RateLimiter(1.05)
    params = {"place_id": place_id, "taxon_id": taxon_ids, "license": LICENCES, "captive": "false",
              "per_page": 200, "order_by": "id", "order": "asc"}
    rl.wait()
    total = s.get(API, params={**params, "per_page": 0}, timeout=60).json()["total_results"]
    bar = tqdm(total=min(total, cap), initial=n, desc=f"iNat {cc}")
    while n < cap:
        rl.wait()
        res = s.get(API, params={**params, "id_above": last_id}, timeout=120).json()["results"]
        if not res:
            break
        n += write_jsonl(out, (slim(o) for o in res), append=True)
        last_id = res[-1]["id"]
        bar.update(len(res))
    bar.close()
    log_download("inaturalist", f"{API}?place_id={place_id}&license={LICENCES}", "CC0-1.0|CC-BY-4.0 (per record)",
                 n, notes=f"country={cc}; total_available={total}; cap={cap}; taxa from configs/aquatic_taxa.yaml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*")
    args = ap.parse_args()
    taxa = ",".join(str(t["inat_id"]) for t in config("aquatic_taxa") if t["inat_id"])
    for cc, c in config("countries").items():
        if args.countries and cc not in args.countries:
            continue
        collect_country(cc, c["inat_place"], c["cap"], taxa)


if __name__ == "__main__":
    main()
