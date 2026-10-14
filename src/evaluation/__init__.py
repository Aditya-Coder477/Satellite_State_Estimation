"""
src/evaluation/__init__.py
==========================
Public API for the Phase 9 Evaluation Framework.

External modules interact with this package exclusively through
:class:`EvaluationEngine`.
"""

from src.evaluation.evaluator import EvaluationEngine

__all__ = ["EvaluationEngine"]
