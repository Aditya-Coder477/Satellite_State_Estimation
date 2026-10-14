"""
src/propagation/sgp4_propagator.py
===================================
SGP4/SDP4 orbit propagator.

Wraps the sgp4 Python library to propagate a satellite from its OMM epoch
forward in configurable time steps, returning a tidy :class:`pandas.DataFrame`
of TEME Cartesian states.

Classes
-------
SGP4Propagator
    Stateful propagator: accepts an OMMParser result, propagates, returns df.

Notes
-----
* All propagation is performed in the TEME (True Equator, Mean Equinox) frame,
  consistent with the SGP4 standard output.  Coordinate frame transformation
  (TEME → GCRF / ECI / ECEF) is deferred to a later pipeline step.
* Time stamps are stored as UTC ISO-8601 strings (``utc_time`` column) and as
  monotonically increasing elapsed seconds from epoch (``elapsed_seconds``).
* The sgp4 library returns position in **km** and velocity in **km/s**.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

from src.propagation.omm_parser import OMMParser
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class SGP4PropagationError(RuntimeError):
    """Raised when the SGP4 integrator returns a non-zero error code."""

    # SGP4 error-code descriptions (from sgp4 source)
    _SGP4_ERRORS: dict[int, str] = {
        1: "mean elements, ecc >= 1.0 or ecc < -0.001",
        2: "mean motion less than 0",
        3: "pert elements, ecc < 0 or ecc > 1",
        4: "semi-latus rectum < 0",
        5: "epoch elements are sub-orbital",
        6: "satellite has decayed",
    }

    def __init__(self, error_code: int, step_idx: int, utc_time: datetime) -> None:
        desc = self._SGP4_ERRORS.get(error_code, "unknown SGP4 error")
        super().__init__(
            f"SGP4 error {error_code} ('{desc}') at step {step_idx} "
            f"(UTC {utc_time.isoformat()})"
        )
        self.error_code = error_code
        self.step_idx = step_idx
        self.utc_time = utc_time


class SGP4Propagator:
    """
    Propagates a satellite orbit using the SGP4/SDP4 model.

    Parameters
    ----------
    parser : OMMParser
        A fully initialised :class:`~src.propagation.omm_parser.OMMParser`.
    duration_days : float
        Total simulation duration [days].
    step_seconds : float
        Propagation time step [seconds].

    Attributes
    ----------
    satrec : Satrec
        The underlying sgp4 satellite record.
    epoch : datetime
        Simulation start time (= OMM epoch, UTC-aware).
    n_steps : int
        Total number of output rows (= ``ceil(duration_days*86400 / step_seconds) + 1``).

    Examples
    --------
    >>> parser = OMMParser("data/raw/omm/GOES16.txt")
    >>> prop   = SGP4Propagator(parser, duration_days=7, step_seconds=60)
    >>> df     = prop.propagate()
    """

    def __init__(
        self,
        parser: OMMParser,
        duration_days: float,
        step_seconds: float,
    ) -> None:
        self._parser = parser
        self._satrec: Satrec = parser.satrec
        self._meta: dict[str, Any] = parser.metadata

        self.duration_days: float = duration_days
        self.step_seconds: float = step_seconds

        self.epoch: datetime = self._meta["epoch"]
        self.n_steps: int = int(round(duration_days * 86_400 / step_seconds)) + 1

        log.info(
            "SGP4Propagator initialised | satellite=%s | norad=%s | epoch=%s | "
            "duration=%.1f days | step=%g s | n_steps=%d",
            self._meta["satellite_name"],
            self._meta["norad_id"],
            self.epoch.isoformat(),
            self.duration_days,
            self.step_seconds,
            self.n_steps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(self) -> pd.DataFrame:
        """
        Run the SGP4 propagation over the configured window.

        Iterates from ``t=0`` to ``t=duration_days*86400`` in steps of
        ``step_seconds``, calls :func:`sgp4.api.Satrec.sgp4` at each epoch
        offset, and assembles a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``utc_time``, ``elapsed_seconds``, ``satellite_name``,
            ``norad_id``, ``x_teme_km``, ``y_teme_km``, ``z_teme_km``,
            ``vx_teme_kms``, ``vy_teme_kms``, ``vz_teme_kms``.

        Raises
        ------
        SGP4PropagationError
            If SGP4 returns a non-zero error code for any time step.
        """
        log.info(
            "Starting propagation: %s → %s",
            self.epoch.isoformat(),
            (self.epoch + timedelta(days=self.duration_days)).isoformat(),
        )
        wall_start = time.perf_counter()

        # Pre-allocate numpy arrays for speed
        n = self.n_steps
        elapsed_arr = np.empty(n, dtype=np.float64)
        pos_arr = np.empty((n, 3), dtype=np.float64)
        vel_arr = np.empty((n, 3), dtype=np.float64)
        utc_arr: list[str] = [""] * n

        step_td = timedelta(seconds=self.step_seconds)

        for i in range(n):
            elapsed_s = i * self.step_seconds
            current_utc: datetime = self.epoch + timedelta(seconds=elapsed_s)

            # Julian date components for sgp4
            jd, fr = jday(
                current_utc.year,
                current_utc.month,
                current_utc.day,
                current_utc.hour,
                current_utc.minute,
                current_utc.second + current_utc.microsecond * 1e-6,
            )

            err, pos, vel = self._satrec.sgp4(jd, fr)

            if err != 0:
                raise SGP4PropagationError(err, i, current_utc)

            elapsed_arr[i] = elapsed_s
            pos_arr[i] = pos   # (x, y, z) km
            vel_arr[i] = vel   # (vx, vy, vz) km/s
            utc_arr[i] = current_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")

        wall_elapsed = time.perf_counter() - wall_start
        log.info(
            "Propagation complete | %d steps | wall-clock %.2f s (%.1f ms/step)",
            n,
            wall_elapsed,
            wall_elapsed / n * 1000,
        )

        df = self._assemble_dataframe(utc_arr, elapsed_arr, pos_arr, vel_arr)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_dataframe(
        self,
        utc_arr: list[str],
        elapsed_arr: np.ndarray,
        pos_arr: np.ndarray,
        vel_arr: np.ndarray,
    ) -> pd.DataFrame:
        """
        Combine raw arrays into a properly typed :class:`~pandas.DataFrame`.

        Parameters
        ----------
        utc_arr : list[str]
            ISO-8601 UTC time strings.
        elapsed_arr : np.ndarray, shape (N,)
            Elapsed seconds from epoch.
        pos_arr : np.ndarray, shape (N, 3)
            TEME position [km].
        vel_arr : np.ndarray, shape (N, 3)
            TEME velocity [km/s].

        Returns
        -------
        pd.DataFrame
        """
        sat_name = self._meta["satellite_name"]
        norad_id = self._meta["norad_id"]

        df = pd.DataFrame(
            {
                "utc_time": pd.array(utc_arr, dtype="string"),
                "elapsed_seconds": elapsed_arr,
                "satellite_name": sat_name,
                "norad_id": norad_id,
                "x_teme_km": pos_arr[:, 0],
                "y_teme_km": pos_arr[:, 1],
                "z_teme_km": pos_arr[:, 2],
                "vx_teme_kms": vel_arr[:, 0],
                "vy_teme_kms": vel_arr[:, 1],
                "vz_teme_kms": vel_arr[:, 2],
            }
        )

        # Cast types explicitly
        df["elapsed_seconds"] = df["elapsed_seconds"].astype("float64")
        df["norad_id"] = df["norad_id"].astype("int64")
        df["satellite_name"] = df["satellite_name"].astype("string")

        return df

    def __repr__(self) -> str:
        return (
            f"SGP4Propagator(satellite={self._meta['satellite_name']!r}, "
            f"duration={self.duration_days}d, step={self.step_seconds}s)"
        )
