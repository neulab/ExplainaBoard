"""Abstract classes for meta-analysis."""
from __future__ import annotations

import abc


class MetaAnalysis:
    """Abstract class for meta-analysis."""

    @abc.abstractmethod
    def run_meta_analysis(self) -> dict | list:
        """Calculate the result of a meta-analysis.

        Returns:
            A dictionary containing the results of the meta-analysis.
        """
        ...
