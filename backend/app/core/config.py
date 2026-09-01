"""
config.py
Application settings using pydantic-settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (C:\Thermoscope-Ai\.env) regardless of cwd
load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=True)
# also try cwd fallback
load_dotenv(override=False)

from pydantic_settings import BaseSettings

# Capture INDIA_BBOX raw from env before pydantic tries JSON decode, then remove so Settings doesn't fail
_raw_bbox = os.getenv("INDIA_BBOX", "")
if "INDIA_BBOX" in os.environ:
    # remove to avoid pydantic JSON parsing error for "68,6,96,36"
    del os.environ["INDIA_BBOX"]
# also handle CORS etc if needed - keep them
_raw_firms_key = os.getenv("FIRMS_MAP_KEY", "")


class Settings(BaseSettings):
    FIRMS_MAP_KEY: str = ""
    DB_URL: str = "sqlite:///./thermoscope.db"
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:8501"]
    INDIA_BBOX: tuple = (68.0, 6.0, 96.0, 36.0)
    CACHE_HOURS: int = 24
    EPS_KM_DEFAULT: float = 2.0
    MIN_SAMPLES_DEFAULT: int = 3

    class Config:
        env_file = None
        env_file_encoding = "utf-8"
        extra = "ignore"


def _parse_bbox(s: str):
    s = s.strip()
    if not s:
        return None
    if s.startswith("[") or s.startswith("("):
        import json, ast
        try:
            return tuple(json.loads(s))
        except Exception:
            try:
                return tuple(ast.literal_eval(s))
            except Exception:
                pass
    if "," in s:
        try:
            return tuple(float(x.strip()) for x in s.split(",") if x.strip())
        except Exception:
            pass
    return None


def get_settings():
    s = Settings()
    # restore FIRMS_MAP_KEY from dotenv if pydantic didn't pick it (env_file relative to cwd may fail)
    if not s.FIRMS_MAP_KEY and _raw_firms_key:
        s.FIRMS_MAP_KEY = _raw_firms_key
    # if .env had INDIA_BBOX, apply it
    if _raw_bbox:
        parsed = _parse_bbox(_raw_bbox)
        if parsed:
            s.INDIA_BBOX = parsed
    else:
        # also try reading .env file directly as fallback
        try:
            env_path = Path(__file__).resolve().parents[3] / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("INDIA_BBOX"):
                        _, val = line.split("=", 1)
                        p = _parse_bbox(val.strip().strip('"').strip("'"))
                        if p:
                            s.INDIA_BBOX = p
                        break
        except Exception:
            pass
    return s
