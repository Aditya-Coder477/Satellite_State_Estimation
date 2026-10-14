"""
src/dataset/collocation_generator.py
====================================
Generates physics collocation points over the simulation time window.

Collocation points are used to evaluate physics constraints in PINN training.
Supports Uniform, Random, and Latin Hypercube Sampling (LHS) strategies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class CollocationStrategy(ABC):
    """Abstract Base Class defining the collocation point generation interface."""

    @abstractmethod
    def generate_points(self, min_time: float, max_time: float, num_points: int, seed: int = 42) -> np.ndarray:
        """
        Generates a 1D array of time points.

        Parameters
        ----------
        min_time : float
            Start of the time interval [seconds].
        max_time : float
            End of the time interval [seconds].
        num_points : int
            Number of points to generate.
        seed : int, optional
            Random seed for stochastic sampling, by default 42.

        Returns
        -------
        np.ndarray
            Flat array of time points of shape (num_points,).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy."""
        pass


class UniformCollocationStrategy(CollocationStrategy):
    """Generates evenly spaced collocation points across the time interval."""

    def generate_points(self, min_time: float, max_time: float, num_points: int, seed: int = 42) -> np.ndarray:
        log.info("Generating %d uniform collocation points from %f to %f", num_points, min_time, max_time)
        return np.linspace(min_time, max_time, num_points)

    @property
    def name(self) -> str:
        return "uniform"


class RandomCollocationStrategy(CollocationStrategy):
    """Generates uniformly distributed random collocation points."""

    def generate_points(self, min_time: float, max_time: float, num_points: int, seed: int = 42) -> np.ndarray:
        log.info("Generating %d random collocation points from %f to %f (seed=%d)", num_points, min_time, max_time, seed)
        rng = np.random.default_rng(seed)
        return rng.uniform(min_time, max_time, num_points)

    @property
    def name(self) -> str:
        return "random"


class LatinHypercubeCollocationStrategy(CollocationStrategy):
    """
    Generates collocation points using 1D Latin Hypercube Sampling (LHS).

    Divides the interval into equal sub-intervals, samples one point uniformly
    from each sub-interval, and shuffles them.
    """

    def generate_points(self, min_time: float, max_time: float, num_points: int, seed: int = 42) -> np.ndarray:
        log.info("Generating %d LHS collocation points from %f to %f (seed=%d)", num_points, min_time, max_time, seed)
        if num_points <= 0:
            return np.array([], dtype=float)

        rng = np.random.default_rng(seed)
        
        # Divide into equal bins
        bins = np.linspace(min_time, max_time, num_points + 1)
        bin_widths = bins[1:] - bins[:-1]
        
        # Sample offset in each bin
        offsets = rng.uniform(0.0, bin_widths, size=num_points)
        points = bins[:-1] + offsets
        
        # Randomly permute the sampled points
        rng.shuffle(points)
        return points

    @property
    def name(self) -> str:
        return "lhs"


class CollocationStrategyFactory:
    """Factory to instantiate the appropriate collocation strategy from config."""

    _STRATEGIES: Dict[str, type[CollocationStrategy]] = {
        "uniform": UniformCollocationStrategy,
        "random": RandomCollocationStrategy,
        "lhs": LatinHypercubeCollocationStrategy,
    }

    @classmethod
    def create(cls, name: str) -> CollocationStrategy:
        """
        Creates an instance of the requested collocation strategy.

        Parameters
        ----------
        name : str
            Name of the strategy (e.g. "uniform", "random", "lhs").

        Returns
        -------
        CollocationStrategy
            Instantiated collocation strategy.

        Raises
        ------
        ValueError
            If the requested strategy name is unknown.
        """
        strategy_class = cls._STRATEGIES.get(name.lower())
        if not strategy_class:
            raise ValueError(
                f"Unknown collocation strategy '{name}'. Available: {list(cls._STRATEGIES.keys())}"
            )
        return strategy_class()


class CollocationGenerator:
    """
    Orchestrates the generation of collocation points.

    Attributes
    ----------
    strategy : CollocationStrategy
        The sampling strategy to use.
    """

    def __init__(self, strategy: CollocationStrategy) -> None:
        self.strategy = strategy

    def generate(
        self,
        min_time: float,
        max_time: float,
        num_points: int,
        normalizer: Any | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generates and structures collocation points.

        Parameters
        ----------
        min_time : float
            Start time [seconds].
        max_time : float
            End time [seconds].
        num_points : int
            Target number of points.
        normalizer : Normalizer, optional
            Normalizer to transform the generated 'time' column into 'normalized_time'.
        seed : int, optional
            Random seed for stochastic strategies, by default 42.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['time', 'normalized_time', 'weight', 'sample_type'].
        """
        points = self.strategy.generate_points(
            min_time=min_time, max_time=max_time, num_points=num_points, seed=seed
        )

        df = pd.DataFrame({
            "time": points,
            "weight": 1.0,
            "sample_type": self.strategy.name,
        })

        # Apply normalization to the time column if a normalizer is provided
        if normalizer is not None:
            # Create a temporary df with the normalizer columns to normalize
            temp_df = pd.DataFrame({"elapsed_seconds": points})
            try:
                norm_df = normalizer.transform(temp_df)
                df["normalized_time"] = norm_df["elapsed_seconds"]
            except Exception as e:
                log.warning("Could not normalize collocation times: %s. Defaulting to raw times.", str(e))
                df["normalized_time"] = points
        else:
            log.warning("No normalizer provided to CollocationGenerator. 'normalized_time' is set equal to 'time'.")
            df["normalized_time"] = points

        return df[["time", "normalized_time", "weight", "sample_type"]]
