"""
src/dataset/dataset_splitter.py
==============================
Provides data splitting strategies for separating datasets into train,
validation, and test splits.

Supports temporal (chronological, non-shuffled) splitting as the default
for PINN/physics simulations, and random splitting for comparison.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class SplitStrategy(ABC):
    """Abstract Base Class defining the dataset splitting strategy interface."""

    @abstractmethod
    def split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the given DataFrame into training, validation, and test subsets.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to split.
        train_ratio : float
            Fraction of the dataset to allocate to the training split.
        val_ratio : float
            Fraction of the dataset to allocate to the validation split.
        test_ratio : float
            Fraction of the dataset to allocate to the testing split.
        seed : int, optional
            Random seed for reproducibility, by default 42.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple containing (train_df, val_df, test_df).
        """
        pass


class TemporalSplitStrategy(SplitStrategy):
    """
    Splits the dataset chronologically without shuffling.

    Ensures that validation and test sets occur strictly after training data
    in time.
    """

    def split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

        # Ensure the dataset is ordered chronologically
        sort_col = "elapsed_seconds" if "elapsed_seconds" in df.columns else "utc_time"
        log.info("Sorting dataset chronologically by '%s' before splitting.", sort_col)
        df_sorted = df.sort_values(by=sort_col, ascending=True).copy()

        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]

        log.info(
            "Temporal split completed: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
            len(train_df),
            (len(train_df) / n) * 100,
            len(val_df),
            (len(val_df) / n) * 100,
            len(test_df),
            (len(test_df) / n) * 100,
        )

        return train_df, val_df, test_df


class RandomSplitStrategy(SplitStrategy):
    """
    Splits the dataset randomly using a given random seed.

    Shuffles the dataset, then partitions it based on split ratios.
    """

    def split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

        log.info("Splitting dataset randomly using seed=%d", seed)
        df_shuffled = df.sample(frac=1.0, random_state=seed).copy()

        n = len(df_shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_shuffled.iloc[:train_end]
        val_df = df_shuffled.iloc[train_end:val_end]
        test_df = df_shuffled.iloc[val_end:]

        log.info(
            "Random split completed: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
            len(train_df),
            (len(train_df) / n) * 100,
            len(val_df),
            (len(val_df) / n) * 100,
            len(test_df),
            (len(test_df) / n) * 100,
        )

        return train_df, val_df, test_df


class SplitStrategyFactory:
    """Factory to instantiate the appropriate splitting strategy from config."""

    _STRATEGIES: Dict[str, type[SplitStrategy]] = {
        "temporal": TemporalSplitStrategy,
        "random": RandomSplitStrategy,
    }

    @classmethod
    def create(cls, name: str) -> SplitStrategy:
        """
        Creates an instance of the requested split strategy.

        Parameters
        ----------
        name : str
            Name of the splitting strategy (e.g. "temporal", "random").

        Returns
        -------
        SplitStrategy
            Instantiated split strategy.

        Raises
        ------
        ValueError
            If the requested strategy name is unknown.
        """
        strategy_class = cls._STRATEGIES.get(name.lower())
        if not strategy_class:
            raise ValueError(
                f"Unknown split strategy '{name}'. Available: {list(cls._STRATEGIES.keys())}"
            )
        return strategy_class()
