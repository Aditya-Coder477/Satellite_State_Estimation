"""
src/dataset/__init__.py
=======================
Phase 6: Dataset Engineering Pipeline.

Provides modules to convert raw simulated observations and ground station tracks into
curated, normalized, split, and feature-engineered datasets with physics collocation
points, fully prepared for PyTorch/PINN model training.

Public API
----------
- DatasetManager         : Controls output directories and manages CSV/JSON I/O.
- DatasetLoader          : Handles raw CSV loading, type checking, and schema validation.
- SplitStrategyFactory   : Creates split strategies (TemporalSplitStrategy, RandomSplitStrategy).
- CollocationGenerator   : Generates physics collocation points (Uniform, Random, LHS).
- Normalizer             : Fits and applies feature scaling (Standard, MinMaxScaler, Identity).
- FeatureEngineer        : Computes periodic, orbital fraction, encoding, and visibility columns.
- DatasetStatistics      : Computes dataset summaries and raw feature stats.
- ObservationDataset     : PyTorch Dataset mapping DataFrames to model training tensors.
- DatasetValidator       : Asserts pipeline output validity (NaNs, ordering, split size bounds).
- DatasetVisualizer      : Renders diagnostic dark-theme PNG plots.
"""

from __future__ import annotations

from src.dataset.collocation_generator import (
    CollocationGenerator,
    CollocationStrategy,
    CollocationStrategyFactory,
    LatinHypercubeCollocationStrategy,
    RandomCollocationStrategy,
    UniformCollocationStrategy,
)
from src.dataset.dataset_loader import DatasetLoader, DatasetSchemaError
from src.dataset.dataset_manager import DatasetManager
from src.dataset.dataset_splitter import (
    RandomSplitStrategy,
    SplitStrategy,
    SplitStrategyFactory,
    TemporalSplitStrategy,
)
from src.dataset.feature_engineering import FeatureEngineer
from src.dataset.normalization import NormalizationError, Normalizer
from src.dataset.pytorch_dataset import ObservationDataset
from src.dataset.statistics import DatasetStatistics
from src.dataset.transition_generator import TransitionGenerator
from src.dataset.validator import DatasetValidator
from src.dataset.visualizer import DatasetVisualizer

__all__ = [
    "DatasetManager",
    "DatasetLoader",
    "DatasetSchemaError",
    "SplitStrategy",
    "TemporalSplitStrategy",
    "RandomSplitStrategy",
    "SplitStrategyFactory",
    "CollocationStrategy",
    "UniformCollocationStrategy",
    "RandomCollocationStrategy",
    "LatinHypercubeCollocationStrategy",
    "CollocationStrategyFactory",
    "CollocationGenerator",
    "Normalizer",
    "NormalizationError",
    "FeatureEngineer",
    "DatasetStatistics",
    "ObservationDataset",
    "TransitionGenerator",
    "DatasetValidator",
    "DatasetVisualizer",
]
