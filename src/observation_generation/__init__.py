"""
src/observation_generation/__init__.py
=======================================
Public API for the Observation Generation package.

Exposes the SensorModel, RA/Dec generator, validator, statistics, visualizer,
and automated scenario generators.
"""

from src.observation_generation.sensor_model import BaseSensorModel, OpticalSensorModel
from src.observation_generation.ra_dec_generator import RADecGenerator
from src.observation_generation.noise_models import NoiseModelFactory
from src.observation_generation.sampling_models import ObservationSampler
from src.observation_generation.observation_quality import QualityEvaluator
from src.observation_generation.scenario_generator import ScenarioGenerator
from src.observation_generation.validator import ObservationValidator
from src.observation_generation.statistics import ObservationStatistics
from src.observation_generation.visualizer import ObservationVisualizer

__all__ = [
    "BaseSensorModel",
    "OpticalSensorModel",
    "RADecGenerator",
    "NoiseModelFactory",
    "ObservationSampler",
    "QualityEvaluator",
    "ScenarioGenerator",
    "ObservationValidator",
    "ObservationStatistics",
    "ObservationVisualizer",
]
