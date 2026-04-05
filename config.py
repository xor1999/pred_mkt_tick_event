"""Project configuration -- loads .env and exposes API keys and paths."""

from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
TELONEX_API_KEY: str = os.getenv("TELONEX_API_KEY", "")
EODHD_API_KEY: str = os.getenv("EODHD_API_KEY", "")
# Kalshi is US-only; leave blank if you don't have access.
# Polymarket via Telonex is the primary prediction market source.
KALSHI_API_KEY: str = os.getenv("KALSHI_API_KEY", "")

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
