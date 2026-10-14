"""
src/propagation/omm_parser.py
=============================
Parses a CSV-formatted OMM (Orbit Mean-elements Message) file and converts
it into a :class:`sgp4.api.Satrec` satellite object plus a metadata dict.

OMM CSV Format (two rows — header + data)
------------------------------------------
The files distributed by Space-Track / CelesTrak as "OMM CSV" look like::

    OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,...
    GOES 16,2016-071A,2026-06-26T05:31:09.471072,1.00272055,...

All field names are drawn from the CCSDS OMM standard (502.0-B-3).

References
----------
* CCSDS 502.0-B-3 Orbit Data Messages
* https://celestrak.org/SOCRATES/query.php
* sgp4 Python library — Brandon Rhodes / Daniel Warner
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sgp4.api import Satrec, WGS72


@dataclass(frozen=True)
class OMMRecord:
    """
    Immutable container for the parsed OMM fields.

    Attributes
    ----------
    object_name : str
        Human-readable satellite name (e.g. ``"GOES 16"``).
    object_id : str
        International Designator (COSPAR ID, e.g. ``"2016-071A"``).
    epoch : datetime
        TLE epoch expressed as a timezone-aware UTC datetime.
    mean_motion : float
        Mean motion [rev/day].
    eccentricity : float
        Orbital eccentricity (dimensionless, 0 ≤ e < 1).
    inclination : float
        Inclination [degrees].
    ra_of_asc_node : float
        Right ascension of ascending node [degrees].
    arg_of_pericenter : float
        Argument of pericenter [degrees].
    mean_anomaly : float
        Mean anomaly at epoch [degrees].
    ephemeris_type : int
        SGP4 ephemeris type (0 = SGP4/SDP4).
    classification_type : str
        ``'U'`` (unclassified), ``'C'`` (classified), ``'S'`` (secret).
    norad_cat_id : int
        NORAD catalog number.
    element_set_no : int
        TLE element set number.
    rev_at_epoch : int
        Revolution number at epoch.
    bstar : float
        BSTAR drag term [1/earth-radii].
    mean_motion_dot : float
        First derivative of mean motion [rev/day²].
    mean_motion_ddot : float
        Second derivative of mean motion [rev/day³].
    raw : dict[str, str]
        Unmodified raw CSV row for reference / debugging.
    """

    object_name: str
    object_id: str
    epoch: datetime
    mean_motion: float
    eccentricity: float
    inclination: float
    ra_of_asc_node: float
    arg_of_pericenter: float
    mean_anomaly: float
    ephemeris_type: int
    classification_type: str
    norad_cat_id: int
    element_set_no: int
    rev_at_epoch: int
    bstar: float
    mean_motion_dot: float
    mean_motion_ddot: float
    raw: dict[str, str] = field(default_factory=dict, compare=False)


class OMMParser:
    """
    Parse a two-row CSV OMM file and build a :class:`~sgp4.api.Satrec` object.

    Parameters
    ----------
    omm_path : str | Path
        Path to the OMM ``.txt`` / ``.csv`` file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not contain exactly one data row, or required
        columns are missing.

    Examples
    --------
    >>> parser = OMMParser("data/raw/omm/GOES16.txt")
    >>> record  = parser.record
    >>> satrec  = parser.satrec
    >>> meta    = parser.metadata
    """

    # Mandatory OMM column names (case-insensitive keys after strip)
    _REQUIRED_COLUMNS: frozenset[str] = frozenset(
        {
            "OBJECT_NAME",
            "OBJECT_ID",
            "EPOCH",
            "MEAN_MOTION",
            "ECCENTRICITY",
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
            "EPHEMERIS_TYPE",
            "CLASSIFICATION_TYPE",
            "NORAD_CAT_ID",
            "ELEMENT_SET_NO",
            "REV_AT_EPOCH",
            "BSTAR",
            "MEAN_MOTION_DOT",
            "MEAN_MOTION_DDOT",
        }
    )

    def __init__(self, omm_path: str | Path) -> None:
        self._path = Path(omm_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"OMM file not found: {self._path}")

        self._record: OMMRecord = self._parse()
        self._satrec: Satrec = self._build_satrec()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def record(self) -> OMMRecord:
        """Parsed :class:`OMMRecord` data object."""
        return self._record

    @property
    def satrec(self) -> Satrec:
        """Initialised :class:`sgp4.api.Satrec` satellite object."""
        return self._satrec

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Convenience dictionary of key satellite identifiers.

        Returns
        -------
        dict with keys: ``satellite_name``, ``norad_id``, ``epoch``,
        ``object_id``, ``omm_path``.
        """
        return {
            "satellite_name": self._record.object_name,
            "norad_id": self._record.norad_cat_id,
            "epoch": self._record.epoch,
            "object_id": self._record.object_id,
            "omm_path": str(self._path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self) -> OMMRecord:
        """
        Read the CSV file, validate columns and row count, return an
        :class:`OMMRecord`.
        """
        with open(self._path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            # Strip BOM / whitespace from field names
            if reader.fieldnames:
                reader.fieldnames = [
                    f.strip().lstrip("\ufeff") for f in reader.fieldnames
                ]

            rows = [row for row in reader if any(v.strip() for v in row.values())]

        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly 1 data row in OMM file, found {len(rows)}: "
                f"{self._path}"
            )

        row: dict[str, str] = {k.strip(): v.strip() for k, v in rows[0].items()}

        missing = self._REQUIRED_COLUMNS - set(row.keys())
        if missing:
            raise ValueError(
                f"OMM file is missing required columns: {sorted(missing)}\n"
                f"  file: {self._path}"
            )

        return OMMRecord(
            object_name=row["OBJECT_NAME"],
            object_id=row["OBJECT_ID"],
            epoch=self._parse_epoch(row["EPOCH"]),
            mean_motion=float(row["MEAN_MOTION"]),
            eccentricity=float(row["ECCENTRICITY"]),
            inclination=float(row["INCLINATION"]),
            ra_of_asc_node=float(row["RA_OF_ASC_NODE"]),
            arg_of_pericenter=float(row["ARG_OF_PERICENTER"]),
            mean_anomaly=float(row["MEAN_ANOMALY"]),
            ephemeris_type=int(row["EPHEMERIS_TYPE"]),
            classification_type=row["CLASSIFICATION_TYPE"],
            norad_cat_id=int(row["NORAD_CAT_ID"]),
            element_set_no=int(row["ELEMENT_SET_NO"]),
            rev_at_epoch=int(row["REV_AT_EPOCH"]),
            bstar=float(row["BSTAR"]),
            mean_motion_dot=float(row["MEAN_MOTION_DOT"]),
            mean_motion_ddot=float(row["MEAN_MOTION_DDOT"]),
            raw=row,
        )

    @staticmethod
    def _parse_epoch(epoch_str: str) -> datetime:
        """
        Parse ISO-8601 epoch string to a UTC-aware :class:`~datetime.datetime`.

        Handles both fractional-second and whole-second variants.

        Parameters
        ----------
        epoch_str : str
            e.g. ``"2026-06-26T05:31:09.471072"``

        Returns
        -------
        datetime
            Timezone-aware UTC datetime.
        """
        epoch_str = epoch_str.strip()
        # Normalise trailing 'Z' → compatible with fromisoformat on Python < 3.11
        epoch_str = epoch_str.rstrip("Z")
        try:
            dt = datetime.fromisoformat(epoch_str)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse OMM epoch '{epoch_str}'. "
                "Expected ISO-8601 format, e.g. '2026-06-26T05:31:09.471072'."
            ) from exc

        # Attach UTC timezone if naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _build_satrec(self) -> Satrec:
        """
        Construct a :class:`~sgp4.api.Satrec` object from the parsed OMM record
        using :func:`sgp4.api.Satrec.twoline2rv` indirectly via the OMM
        initialiser.

        The sgp4 library's ``Satrec.twoline2rv`` method is TLE-oriented.
        We use ``sgp4.conveniences.sat_epoch_datetime`` approach via
        ``sgp4.exporter`` / ``sgp4.omm`` to correctly initialise from OMM.

        Returns
        -------
        Satrec
            Fully initialised SGP4/SDP4 satellite record.
        """
        try:
            from sgp4 import omm as sgp4_omm
        except ImportError as exc:
            raise ImportError(
                "The 'sgp4' package is required. Install with: pip install sgp4"
            ) from exc

        rec = self._record

        # Build the field dict expected by sgp4.omm.initialize()
        omm_fields: dict[str, Any] = {
            "OBJECT_NAME": rec.object_name,
            "OBJECT_ID": rec.object_id,
            "EPOCH": rec.epoch.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "MEAN_MOTION": str(rec.mean_motion),
            "ECCENTRICITY": str(rec.eccentricity),
            "INCLINATION": str(rec.inclination),
            "RA_OF_ASC_NODE": str(rec.ra_of_asc_node),
            "ARG_OF_PERICENTER": str(rec.arg_of_pericenter),
            "MEAN_ANOMALY": str(rec.mean_anomaly),
            "EPHEMERIS_TYPE": str(rec.ephemeris_type),
            "CLASSIFICATION_TYPE": rec.classification_type,
            "NORAD_CAT_ID": str(rec.norad_cat_id),
            "ELEMENT_SET_NO": str(rec.element_set_no),
            "REV_AT_EPOCH": str(rec.rev_at_epoch),
            "BSTAR": str(rec.bstar),
            "MEAN_MOTION_DOT": str(rec.mean_motion_dot),
            "MEAN_MOTION_DDOT": str(rec.mean_motion_ddot),
        }

        satrec = Satrec()
        sgp4_omm.initialize(satrec, omm_fields)
        return satrec

    def __repr__(self) -> str:
        return (
            f"OMMParser(path={str(self._path)!r}, "
            f"satellite={self._record.object_name!r}, "
            f"norad_id={self._record.norad_cat_id})"
        )
