"""Platform-specific raw -> Record mappers. Each yields dicts with the Record fields plus a few
``_``-prefixed helper keys (locality, obligate) that build.py uses for the water filter and then drops.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Iterator

from cswo import RAW
from cswo.collect.common import config, read_jsonl
from cswo.harmonize.anonymize import coarsen, hash_user, scrub_text
from cswo.licence import normalise, redistributable

# --------------------------------------------------------------------------------------------- helpers

_BOILERPLATE = [
    re.compile(r"^\s*Quality assessment:\s*[^|]*\|?\s*", re.I),       # Laji.fi prefix
    # Tiira (Finnish bird database) exports: keep the observer's note, drop the wrapper and the trailing dump,
    # which holds exact coordinates and observer names
    re.compile(r"/?\s*tallennettu Tiiraan:.*$", re.I | re.S),
    re.compile(r"^\s*(ali)?havainnon lisätiedot:\s*", re.I),
    re.compile(r"^\s*Tiira-erittely:\s*", re.I),
    re.compile(r"Keywords:\s*\{[^}]*\}\s*\|?\s*", re.I),               # Laji.fi tag wrapper
    re.compile(r"^\s*Organism Quantity represents.*$", re.I | re.S),   # survey template
    re.compile(r"(?<!\w)\w+ rapportkommitt[ée].*$", re.I | re.S),         # Artportalen import stamp (+ person names)
    re.compile(r"https?://\S+"),                                       # links carry no observation text
]
MIN_NOTE_CHARS = 2                                                      # Laji.fi "p" (pair) codes are dropped


def clean_text(t: str | None) -> str | None:
    if not t:
        return None
    for pat in _BOILERPLATE:
        t = pat.sub("", t)
    t = re.sub(r"<[^>]+>", " ", t)                 # iNat descriptions may contain HTML
    t = re.sub(r"\s+", " ", t).strip(" |;,.-")
    return scrub_text(t) if len(t) >= MIN_NOTE_CHARS else None


def forel_ule_bin3(fu: int | None) -> str | None:
    """FU 1-5 blue (clear, oligotrophic), 6-13 green (productive), 14-21 brown/yellow (humic, sediment)."""
    if fu is None or not 1 <= fu <= 21:
        return None
    return "blue" if fu <= 5 else "green" if fu <= 13 else "brown"


def _obligate_ids(key: str) -> tuple[set, set]:
    """iNat taxon ids, or GBIF (rank, name) pairs, of obligate and facultative aquatic taxa."""
    taxa = config("aquatic_taxa")
    if key == "inat":
        ids = lambda ob: {t["inat_id"] for t in taxa if t["obligate"] == ob and t["inat_id"]}
    else:
        ids = lambda ob: {(x["rank"], x["name"]) for t in taxa if t["obligate"] == ob for x in t["gbif_xr"]}
    return ids(True), ids(False)


# --------------------------------------------------------------------------------------------- EyeOnWater

def eyeonwater() -> Iterator[dict]:
    path = max((RAW / "eyeonwater").glob("eyeonwater_full_*.json"))
    for r in json.loads(path.read_text(encoding="utf-8")):
        img = (r.get("image") or {}).get("image")
        if not img:
            continue
        native = Path(img).stem
        w, loc = r.get("water") or {}, r.get("location") or {}
        fu = w.get("fu_value") or w.get("fu_processed")
        fu = int(fu) if fu else None
        sd = w.get("sd_depth")
        yield {
            "id": f"eyeonwater:{native}", "source_platform": "eyeonwater",
            "text_original": None,
            "media": [{"url": img.replace("http://", "https://"), "licence": "EyeOnWater-open", "redistributable": False}],
            "timestamp": (r["image"].get("date_photo") or "").replace(" ", "T") or None,
            "lat": coarsen(loc.get("lat")), "lon": coarsen(loc.get("lng")),
            "labels": {"forel_ule": fu if fu and 1 <= fu <= 21 else None, "forel_ule_bin3": forel_ule_bin3(fu),
                       "secchi_depth_m": sd if isinstance(sd, (int, float)) and sd > 0 else None,
                       "temperature_c": w.get("p_temperature")},
            "licence": "EyeOnWater-open", "provenance_url": img,
            # EyeOnWater "nickname" is often an e-mail address: hash the stable numeric code instead when present
            "user_hash": hash_user("eyeonwater", (r.get("user") or {}).get("user_n_code") or (r.get("user") or {}).get("nickname")),
            "water_body_type": "unknown", "water_evidence": "platform:water_only",
        }


# --------------------------------------------------------------------------------------------- iNaturalist

def inaturalist() -> Iterator[dict]:
    obligate, facultative = _obligate_ids("inat")
    for path in sorted((RAW / "inaturalist").glob("inat_*.jsonl")):
        cc = path.stem.split("_")[1]
        for o in read_jsonl(path):
            lic = normalise(o.get("license_code"))
            if not redistributable(lic):
                continue
            ancestors = set((o.get("taxon") or {}).get("ancestor_ids") or [])
            lat = lon = None
            if o.get("location"):
                lat, lon = map(float, o["location"].split(","))
            photos = [{"url": p["url"].replace("/square.", "/medium."), "licence": normalise(p.get("license_code")),
                       "redistributable": redistributable(normalise(p.get("license_code")))}
                      for p in o.get("photos", []) if p.get("url")]
            yield {
                "id": f"inaturalist:{o['id']}", "source_platform": "inaturalist",
                "text_original": clean_text(o.get("description")),
                "media": photos,
                "timestamp": o.get("time_observed_at") or o.get("observed_on"),
                "lat": coarsen(lat), "lon": coarsen(lon), "country": cc,
                "labels": {"species": (o.get("taxon") or {}).get("name")},
                "licence": lic, "provenance_url": o.get("uri") or f"https://www.inaturalist.org/observations/{o['id']}",
                "user_hash": hash_user("inaturalist", o.get("user_id")),
                "_locality": o.get("place_guess"), "_habitat": None,
                "_obligate": bool(ancestors & obligate), "_aquatic": bool(ancestors & (obligate | facultative)),
                "_country_lang": cc,
            }


# --------------------------------------------------------------------------------------------- GBIF portals

def _gbif_rows(path: Path) -> Iterator[dict]:
    """Rows from a SQL_TSV_ZIP or a DWCA zip, with lower-cased column names."""
    with zipfile.ZipFile(path) as z:
        name = "occurrence.txt" if "occurrence.txt" in z.namelist() else next(
            n for n in z.namelist() if n.endswith((".tsv", ".csv", ".txt")))
        with z.open(name) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                yield {k.lower().strip("`"): (v or None) for k, v in row.items() if k}


def gbif() -> Iterator[dict]:
    obligate, facultative = _obligate_ids("gbif")
    datasets = {d["key"]: d for d in config("gbif_datasets")}
    seen: set[str] = set()
    for path in sorted((RAW / "gbif").glob("*.zip")):
        for r in _gbif_rows(path):
            gid = r.get("gbifid")
            ds = datasets.get(r.get("datasetkey") or "")
            if not gid or gid in seen or ds is None:
                continue
            seen.add(gid)
            lic = normalise(r.get("license"))
            text = clean_text(r.get("occurrenceremarks"))
            if not redistributable(lic) or not text:
                continue
            ranks = {(c, r[c]) for c in ("phylum", "class", "order", "family", "subfamily", "genus") if r.get(c)}
            media = [{"url": u, "licence": lic, "redistributable": False}
                     for u in re.split(r"[|;, ]+", r.get("v_associatedmedia") or r.get("associatedmedia") or "")
                     if u.startswith("http")]
            ref = r.get("references")
            yield {
                "id": f"{ds['platform']}:{gid}", "source_platform": ds["platform"], "source_dataset": ds["key"],
                "text_original": text, "media": media,
                "timestamp": r.get("eventdate"),
                "lat": coarsen(float(r["decimallatitude"])) if r.get("decimallatitude") else None,
                "lon": coarsen(float(r["decimallongitude"])) if r.get("decimallongitude") else None,
                "country": r.get("countrycode"),
                "labels": {"species": r.get("species") or r.get("scientificname"), "habitat": r.get("habitat")},
                "licence": lic,
                "provenance_url": ref if ref and ref.startswith("http") else f"https://www.gbif.org/occurrence/{gid}",
                "user_hash": hash_user(ds["platform"], r.get("recordedby")),
                "_locality": " ".join(filter(None, [r.get("waterbody"), r.get("locality")])),
                "_habitat": r.get("habitat"),
                "_obligate": bool(ranks & obligate), "_aquatic": bool(ranks & (obligate | facultative)),
                "_country_lang": r.get("countrycode"),
            }


# --------------------------------------------------------------------------------------------- CrowdWater

# Spotteron CSV export (columns checked on the 2026-10-25 export, 86 columns, upper case).
# Soil moisture is not a water-body observation; other categories are streams, gauges, lakes or litter at water.
CW_KEEP = {"temporary stream": "stream", "virtual scale": "river", "physical scale": "river",
           "stream type": "stream", "standing water type": "lake", "plastic pollution": "unknown"}
# Structured observations kept as weak labels / gold hints (spec 8), stored under labels.extra.
CW_EXTRA = {"flow_type": "FLOW_TYPE", "snow_ice": "SNOW_ICE_PRESENT", "foam": "FOAM", "algae": "ALGAE",
            "odour": "ODOR", "litter": "LITTER_IN_WATER", "clarity": "WATER_CLARITY",
            "colour": "STREAMTYPE_WATERCOLOR", "lake_colour": "LAKE_WATER_COLOUR", "lake_transparency": "LAKE_TRANSPARENCY",
            "plastic_amount": "PP_AMOUNT", "stream_name": "STREAMTYPE_NAME"}


def _cw_level(v: str | None) -> str | None:
    """Virtual staff gauge class relative to the reference photo: 'plus 2' -> '+2', 'minus 1' -> '-1'."""
    m = re.fullmatch(r"(plus|minus)\s*(\d+)", (v or "").strip().lower())
    return None if not m else ("+" if m[1] == "plus" else "-") + m[2]


def crowdwater() -> Iterator[dict]:
    lic = normalise(config("manual_sources")["crowdwater"]["licence"])
    if not redistributable(lic):
        print("CrowdWater skipped: set its licence in configs/manual_sources.yaml")
        return
    for path in sorted((RAW / "crowdwater").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                row = {k.upper().strip(): (v.strip() if isinstance(v, str) else None) or None
                       for k, v in row.items() if k}
                native, category = row.get("ID"), (row.get("CATEGORY") or "").lower()
                if not native or category not in CW_KEEP:      # also drops a few rows broken by bad quoting
                    continue
                wbt = "lake" if row.get("PRESELECT_WATER_TYPE") == "Lake" else CW_KEEP[category]
                img = row.get("IMAGE")
                extra = {k: row[c] for k, c in CW_EXTRA.items() if row.get(c) and row[c] != "false"}
                yield {
                    "id": f"crowdwater:{native}", "source_platform": "crowdwater",
                    "text_original": clean_text(row.get("DESCRIPTION")),
                    "media": [{"url": img, "licence": lic, "redistributable": False}] if img else [],
                    "timestamp": row.get("SPOTTED_AT") or row.get("CREATED_AT"),
                    "lat": coarsen(float(row["LATITUDE"])) if row.get("LATITUDE") else None,
                    "lon": coarsen(float(row["LONGITUDE"])) if row.get("LONGITUDE") else None,
                    "labels": {"water_level_class": _cw_level(row.get("WATER_LEVEL")),
                               "category": category, "series_id": row.get("ROOT_ID"), "extra": extra or None},
                    "licence": lic,
                    "provenance_url": f"https://www.spotteron.com/crowdwater/spots/{native}",
                    "user_hash": None,             # the export carries no user identifier
                    "water_body_type": wbt, "water_evidence": f"platform:crowdwater:{category}",
                }
