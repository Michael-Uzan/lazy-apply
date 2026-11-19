"""The evaluation layer: an EvidencePack in, a grounded FitReport out.

``build_evaluator`` is the only place that decides whether judgments come from
Claude or from the offline fake.
"""

from __future__ import annotations

from lazyapply_core.config import Settings
from lazyapply_core.evaluate.base import EvaluationError, Evaluator
from lazyapply_core.evaluate.engine import evaluate_fit
from lazyapply_core.evaluate.verifier import CitationError

__all__ = [
    "CitationError",
    "EvaluationError",
    "Evaluator",
    "build_evaluator",
    "evaluate_fit",
]


def build_evaluator(settings: Settings, *, fake: bool = False) -> Evaluator:
    if fake:
        from lazyapply_core.evaluate.fake import ScoreBasedEvaluator

        return ScoreBasedEvaluator()

    from lazyapply_core.evaluate.claude import ClaudeEvaluator

    return ClaudeEvaluator(api_key=settings.anthropic_api_key, model=settings.judgment_model)
