"""Remove personal data: hash user identifiers, coarsen coordinates, scrub e-mails from text."""
from __future__ import annotations

import hashlib
import os
import re

from dotenv import load_dotenv

load_dotenv()

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_USER = re.compile(r"(inaturalist\.org/people/|observation\.org/users/)\S+", re.I)
# Decimal coordinate pairs typed into notes ("Lat/Lon: 56.98194, 24.26111") would undo the 1 km coarsening.
_COORD_PAIR = re.compile(r"(?<![\d.])-?\d{1,3}[.,]\d{3,}\s*[,;/ ]\s*-?\d{1,3}[.,]\d{3,}(?!\d)")


def _salt() -> str:
    salt = os.environ.get("CSWO_SALT")
    if not salt:
        raise RuntimeError("Set CSWO_SALT in .env (see .env.example) before harmonizing.")
    return salt


def hash_user(platform: str, user_id: object) -> str | None:
    """Salted SHA-256, truncated. The same user on the same platform maps to the same hash (needed for Gini)."""
    if user_id is None or str(user_id).strip() == "":
        return None
    h = hashlib.sha256(f"{_salt()}|{platform}|{user_id}".encode()).hexdigest()
    return h[:16]


def coarsen(x: float | None, decimals: int = 2) -> float | None:
    """0.01 degree ~ 1.1 km in latitude, less in longitude at high latitude; conservative for spec's 1 km."""
    return None if x is None else round(float(x), decimals)


def scrub_text(text: str | None) -> str | None:
    if not text:
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _COORD_PAIR.sub("[COORD]", text)
    return _URL_USER.sub("[USER_URL]", text)
