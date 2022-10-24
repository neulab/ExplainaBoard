"""Package definition for explainaboard.meta_analysis."""

from __future__ import annotations

from explainaboard.meta_analyses.rank_flipping import RankFlippingMetaAnalysis
from explainaboard.meta_analyses.ranking import RankingMetaAnalysis

__all__ = ["RankFlippingMetaAnalysis", "RankingMetaAnalysis"]
