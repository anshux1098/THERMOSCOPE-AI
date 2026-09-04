"""
label_aggregator.py
Aggregation and Consensus Layer for Weak Supervision in THERMOSCOPE-AI (SIH26162).

Architecture:
- Collects independent votes from Labeling Functions (LFs).
- Filters out ABSTAIN (None) decisions.
- Aggregates non-abstaining votes via majority rule consensus.
- Returns 'unclassified' as a safe default fallback when all LFs abstain or upon an unresolved tie.
- Keeps classification decision logic separate from individual LF heuristic definitions.
"""
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from app.core.constants import CLASS_LABELS

# Canonical fallback class
UNCLASSIFIED: str = "unclassified"


def apply_labeling_functions(
    record: Any,
    lfs: Optional[Sequence[Callable[[Any], Optional[str]]]] = None,
) -> Dict[str, Optional[str]]:
    """
    Apply a sequence of labeling functions to a single hotspot record.

    Args:
        record: Hotspot representation (dict, Pydantic model, or Pandas Series).
        lfs: Sequence of labeling function callables. If None, imports ALL_LABELING_FUNCTIONS.

    Returns:
        Dict mapping labeling function name -> voted_class_or_None (ABSTAIN).
    """
    if lfs is None:
        from app.intelligence.labeling_functions import ALL_LABELING_FUNCTIONS
        lfs = ALL_LABELING_FUNCTIONS

    return {lf.__name__: lf(record) for lf in lfs}


def aggregate_votes(
    votes: Union[Dict[str, Optional[str]], Sequence[Optional[str]]],
    fallback_label: str = UNCLASSIFIED,
) -> str:
    """
    Aggregate LF votes into a single consensus prediction using majority voting.

    Rules:
    1. Filters out ABSTAIN (None) votes.
    2. If all LFs abstain -> returns fallback_label ('unclassified').
    3. Returns the most frequent non-abstain class.
    4. On a tie, selects the top voted class if valid, or falls back safely.

    Args:
        votes: Either a dict of {lf_name: vote} or a sequence of votes.
        fallback_label: Canonical label when no consensus is reached (default: 'unclassified').

    Returns:
        Consensus class string from CLASS_LABELS.
    """
    if isinstance(votes, dict):
        active_votes = [v for v in votes.values() if v is not None]
    elif isinstance(votes, (list, tuple)):
        active_votes = [v for v in votes if v is not None]
    else:
        active_votes = []

    if not active_votes:
        return fallback_label

    counts = Counter(active_votes)
    top_class, top_count = counts.most_common(1)[0]

    # Validate that top_class is recognized in canonical taxonomy
    if top_class in CLASS_LABELS:
        return top_class

    return fallback_label


def compute_vote_summary(votes: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Generate an explainable summary of votes cast by labeling functions.

    Returns:
        Dictionary containing consensus, active vote count, abstention count, and breakdown.
    """
    active = {k: v for k, v in votes.items() if v is not None}
    abstains = [k for k, v in votes.items() if v is None]
    consensus = aggregate_votes(votes)

    return {
        "consensus": consensus,
        "total_lfs": len(votes),
        "active_votes_count": len(active),
        "abstain_count": len(abstains),
        "vote_breakdown": Counter(active.values()),
        "active_lfs": active,
    }
