"""Common record schema shared by every platform (spec 2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Platform = Literal[
    "eyeonwater", "inaturalist", "crowdwater", "waterrangers",
    "artportalen", "laji_fi", "eelurikkus", "observation_org", "dabasdati", "gbif_other",
]


class Labels(BaseModel):
    """Structured labels as reported by the platform. Unknown = None."""
    water_level_class: str | None = None      # CrowdWater virtual staff gauge class
    forel_ule: int | None = Field(None, ge=1, le=21)
    forel_ule_bin3: Literal["blue", "green", "brown"] | None = None
    secchi_depth_m: float | None = None
    turbidity: str | None = None
    temperature_c: float | None = None
    species: str | None = None                 # scientific name
    habitat: str | None = None                 # free-text or coded habitat (Artportalen, GBIF habitat)
    category: str | None = None                # platform observation type (CrowdWater: virtual scale, temporary stream, ...)
    series_id: str | None = None               # repeated observations of one site (CrowdWater ROOT_ID)
    extra: dict[str, str] | None = None        # other platform-specific structured fields, verbatim


class Media(BaseModel):
    url: str
    licence: str | None = None
    redistributable: bool = False              # False -> release URL only, never the file


class Record(BaseModel):
    id: str                                    # "<platform>:<native id>"
    source_platform: Platform
    source_dataset: str | None = None          # e.g. GBIF datasetKey
    language: str | None = None                # ISO 639-1 from language identification, "und" if unknown
    language_confidence: float | None = None
    languages_all: list[str] | None = None     # set only for mixed-language notes
    text_original: str | None = None
    text_en: str | None = None
    translation_model: str | None = None       # None when the original is English
    translation_qc: str | None = None          # "ok" or "rejected:<empty|loop|omission|hallucination>" (mt.qc_flag)
    media: list[Media] = []
    timestamp: str | None = None               # ISO 8601
    lat: float | None = None                   # coarsened to 0.01 degrees
    lon: float | None = None
    country: str | None = None                 # ISO 3166-1 alpha-2
    water_body_type: str | None = None         # lake / river / stream / coast / sea / pond / unknown
    water_evidence: str | None = None          # why the record passed the water filter
    labels: Labels = Labels()
    licence: str                               # normalised SPDX-like code, e.g. CC0-1.0, CC-BY-4.0
    provenance_url: str
    user_hash: str | None = None               # salted hash, never the raw user id
    persons_masked: int | None = None          # personal names replaced by [PERSON] (original + translation)

    @field_validator("lat", "lon")
    @classmethod
    def _coarse(cls, v: float | None) -> float | None:
        return None if v is None else round(v, 2)

    @property
    def has_text(self) -> bool:
        return bool(self.text_original and self.text_original.strip())

    @property
    def has_image(self) -> bool:
        return bool(self.media)
