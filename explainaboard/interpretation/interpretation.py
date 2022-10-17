"""Basic classes to implement performance interpretation and suggestion."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class InterpretationStats:
    """A base class specifying the statistics of an interpretation."""

    ...


@dataclass
class InterpretationObservation:
    """A base class specifying the observations of an interpretation.

    Attributes:
        keywords: brief keywords to summarize the observation
        content: concrete text to describe the observation

    """

    keywords: str
    content: str


@dataclass
class InterpretationSuggestion:
    """A base class specifying the suggestions of an interpretation.

    Attributes:
        keywords: brief keywords to summarize the suggestion
        content: concrete text to describe the suggestion

    """

    keywords: str
    content: str


@dataclass
class Interpretation:
    """A base class specifying the overall results of an interpretation."""

    # For analysis such as BucketAnalysis, the observation is metric-dependent, so here
    # a dictionary is used to store observations for corresponding metrics.

    observations: Mapping[str, list[InterpretationObservation]] | list[
        InterpretationObservation
    ]
    suggestions: Mapping[str, list[InterpretationSuggestion]] | list[
        InterpretationSuggestion
    ]


class Interpreter(metaclass=abc.ABCMeta):
    """A super-class as a controller to perform the interpretation process.

    The actual details of the result will be implemented by the inheriting class.

    """

    @abc.abstractmethod
    def cal_stats(self) -> Any:
        """Calculate statistics for an interpretation."""
        ...

    @abc.abstractmethod
    def generate_observations(
        self,
        interpretation_stats: Mapping[str, InterpretationStats] | InterpretationStats,
    ) -> Any:
        """Generate observations for an interpretation."""
        ...

    @abc.abstractmethod
    def generate_suggestions(
        self,
        observations: Mapping[str, list[InterpretationObservation]]
        | list[InterpretationObservation],
    ) -> Any:
        """Generate suggestions for an interpretation."""
        ...

    def perform(self) -> Interpretation:
        """Perform actual interpretation process."""
        stat = self.cal_stats()
        observations = self.generate_observations(stat)
        suggestions = self.generate_suggestions(observations)

        return Interpretation(observations=observations, suggestions=suggestions)
