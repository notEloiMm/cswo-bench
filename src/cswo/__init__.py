"""CSWO-Bench: citizen science water observations corpus."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
