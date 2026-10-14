"""
src/coordinate_systems/__init__.py
==================================
Coordinate systems transformation package.

Exposes the public APIs to perform TEME -> GCRS -> ITRS transformations,
validate transformations, visualize results, and compute Ground Station local ENU frames.
"""

from src.coordinate_systems.coordinate_transformer import (
    CoordinateTransformer,
    transform_dataframe,
    transform_single_state,
)
from src.coordinate_systems.teme_to_gcrs import teme_to_gcrs_dataframe, teme_to_gcrs_vector
from src.coordinate_systems.gcrs_to_itrs import gcrs_to_itrs_dataframe, gcrs_to_itrs_vector
from src.coordinate_systems.station_frame import GroundStation, GroundStationFrame, StationTransformer
from src.coordinate_systems.astronomy_utils import AstronomyTimeConverter
from src.coordinate_systems.validator import CoordinateTransformationValidator, ValidationResult
from src.coordinate_systems.visualizer import CoordinateVisualizer

__all__ = [
    "CoordinateTransformer",
    "transform_dataframe",
    "transform_single_state",
    "teme_to_gcrs_dataframe",
    "teme_to_gcrs_vector",
    "gcrs_to_itrs_dataframe",
    "gcrs_to_itrs_vector",
    "GroundStation",
    "GroundStationFrame",
    "StationTransformer",
    "AstronomyTimeConverter",
    "CoordinateTransformationValidator",
    "ValidationResult",
    "CoordinateVisualizer",
]
