"""
constants.py
Shared constants for THERMOSCOPE-AI.
"""
SITE_TYPES = [
    "refinery",
    "power_plant",
    "factory",
    "industrial_zone",
    "oil_gas",
    "volcano",
    "mining",
    "other_industrial",
    "power_infrastructure",
]

# Re-exported from classifier/base.py for convenience
from app.intelligence.classifier.base import CLASS_LABELS, CLASS_COLORS
