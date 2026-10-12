"""Read SWC files into `Skeleton`s, with numpy only.

Comment lines and trailing ``#`` comments are ignored. A Janelia/Horta
``# OFFSET X Y Z`` header before the data is folded into the coordinates.
"""
import os
from pathlib import Path

import numpy as np

from neuron_ng.core.model import Skeleton

SWC_COLUMNS = ("node_id", "compartment", "x", "y", "z", "radius", "parent")


class SWCError(ValueError):
    """Raised when an SWC file cannot be turned into a skeleton."""


def read_swc_table(path):
    """Return ``{column: (N,) float array}`` for the SWC at ``path``.

    Values that are not numbers, and missing trailing columns, become NaN.
    """
    path = Path(path)
    offset = np.zeros(3)
    rows = []
    in_header = True
    with open(path, "r") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if in_header and stripped.startswith("# OFFSET"):
                offset = _parse_offset(stripped, path)
            data = stripped.split("#", 1)[0].split()
            if not data:
                continue
            in_header = False
            row = [_to_float(v) for v in data[:len(SWC_COLUMNS)]]
            rows.append(row + [np.nan] * (len(SWC_COLUMNS) - len(row)))
    table = np.array(rows, dtype=float).reshape(-1, len(SWC_COLUMNS))
    columns = {name: table[:, i] for i, name in enumerate(SWC_COLUMNS)}
    for i, axis in enumerate("xyz"):
        columns[axis] = columns[axis] + offset[i]
    return columns


def read_swc(path):
    """Return the SWC at ``path`` as a `Skeleton`, or raise `SWCError`."""
    columns = read_swc_table(path)
    for name in ("node_id", "compartment", "parent"):
        values = columns[name]
        if not np.all(np.isfinite(values)) or np.any(values != np.round(values)):
            raise SWCError(f"{path}: column {name!r} must hold integers")
    if not np.all(np.isfinite(columns["radius"])):
        raise SWCError(f"{path}: column 'radius' must hold numbers")
    xyz = np.stack([columns["x"], columns["y"], columns["z"]], axis=1)
    if not np.all(np.isfinite(xyz)):
        raise SWCError(f"{path}: coordinates must be finite numbers")
    return Skeleton.from_swc_arrays(
        node_id=columns["node_id"], compartment=columns["compartment"], xyz=xyz,
        radius=columns["radius"], parent_id=columns["parent"],
    )


def swc_label(path):
    """Default label for an SWC: its file name without the suffix."""
    return Path(os.fspath(path)).stem


def _parse_offset(header, path):
    parts = header.split()
    try:
        if len(parts) != 5:
            raise ValueError
        return np.array([float(v) for v in parts[2:5]])
    except ValueError:
        raise SWCError(
            f"{path}: invalid OFFSET header {header!r}; expected '# OFFSET X Y Z'"
        ) from None


def _to_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan
