"""Read standard_morph QC reports (``RunReport.to_dict()`` JSON).

This is the only module that knows the report's field names. It reads the
JSON, not standard_morph objects, so reports from any run (or any tool that
writes the same shape) can be shown without installing standard_morph.

The fields used are ``schema_version``, ``input_ref``, ``morphology_evaluated``,
and, for each result in ``integrity_results`` and ``results``, its ``name``,
``status``, ``flagged_node_ids`` and ``flagged_node_coordinates`` (parallel
lists). Other keys are ignored, so a report wrapped with extra blocks (e.g. the
swc_qc capsule's ``validation``) reads the same.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Major versions of the report schema this adapter understands.
SUPPORTED_SCHEMA_MAJORS = ("1",)
#: Result statuses whose flagged nodes are shown. A passing result has no
#: flagged nodes; a skipped one never ran.
FLAG_STATUSES = ("fail", "review", "error")
#: Every status a flagged node can have, as stored in annotation properties.
STATUS_LABELS = ("fail", "review", "error")


class ReportError(ValueError):
    """Raised when the input is not a standard_morph QC report."""


@dataclass
class FlaggedNode:
    node_id: int
    xyz: Tuple[float, float, float]
    status: str


@dataclass
class MetricOutcome:
    name: str
    status: str
    flagged_node_ids: List[int] = field(default_factory=list)
    flagged_node_coordinates: List[List[float]] = field(default_factory=list)


@dataclass
class QCReport:
    """The parts of a QC report that views need."""
    schema_version: Optional[str]
    input_ref: Optional[str]
    #: False when a blocking integrity check failed, so no morphology was built.
    morphology_evaluated: bool
    #: Integrity results first, then morphology results, in report order.
    results: List[MetricOutcome] = field(default_factory=list)

    def flagged_nodes(self, statuses=FLAG_STATUSES, xyz_by_id=None):
        """Return ``{metric: [FlaggedNode]}`` for every result with a status in
        ``statuses`` that flagged nodes.

        Some results (notably the integrity checks) report node ids without
        coordinates. Given ``xyz_by_id`` (e.g. ``Skeleton.xyz_by_id()``), those
        nodes are placed by id. Nodes that still have no finite coordinates
        cannot be placed, so they are left out.
        """
        xyz_by_id = xyz_by_id or {}
        flags = {}
        for result in self.results:
            if result.status not in statuses or not result.flagged_node_ids:
                continue
            coordinates = list(result.flagged_node_coordinates)
            nodes = []
            for i, node_id in enumerate(result.flagged_node_ids):
                xyz = coordinates[i] if i < len(coordinates) else xyz_by_id.get(int(node_id), ())
                xyz = tuple(float(v) for v in xyz)
                if len(xyz) == 3 and np.all(np.isfinite(xyz)):
                    nodes.append(FlaggedNode(int(node_id), xyz, result.status))
            if nodes:
                flags[result.name] = nodes
        return flags


def load_report(report):
    """Return a `QCReport` from a report dict, a path to its JSON, or any
    object with ``to_dict()`` (such as a standard_morph ``RunReport``)."""
    if isinstance(report, (str, os.PathLike)):
        path = Path(report)
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ReportError(f"{path} is not valid JSON: {exc}") from None
    elif hasattr(report, "to_dict"):
        report = report.to_dict()
    if not isinstance(report, dict):
        raise ReportError(f"Expected a QC report dict; got {type(report).__name__}")

    missing = [key for key in ("integrity_results", "results") if key not in report]
    if missing:
        raise ReportError(f"Not a standard_morph QC report: missing {missing}")
    version = report.get("schema_version")
    if version is None or str(version).split(".")[0] not in SUPPORTED_SCHEMA_MAJORS:
        logger.warning(
            "QC report schema_version %r is not one of the supported major versions %s; "
            "reading it anyway", version, list(SUPPORTED_SCHEMA_MAJORS),
        )

    results = []
    for key in ("integrity_results", "results"):
        for raw in report[key]:
            results.append(MetricOutcome(
                name=raw["name"],
                status=raw.get("status", "pass"),
                flagged_node_ids=[int(v) for v in raw.get("flagged_node_ids") or []],
                flagged_node_coordinates=list(raw.get("flagged_node_coordinates") or []),
            ))
    return QCReport(
        schema_version=version,
        input_ref=report.get("input_ref"),
        morphology_evaluated=bool(report.get("morphology_evaluated", True)),
        results=results,
    )


__all__ = [
    "FLAG_STATUSES",
    "STATUS_LABELS",
    "SUPPORTED_SCHEMA_MAJORS",
    "FlaggedNode",
    "MetricOutcome",
    "QCReport",
    "ReportError",
    "load_report",
]
