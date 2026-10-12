"""QC review views: the nodes a standard_morph QC run flagged, on the neurons.

Typical batch use::

    from neuron_ng.recipes.qc_review import QCReviewExporter
    from neuron_ng.storage import upload_dir

    exporter = QCReviewExporter(scale_um=(0.748, 0.748, 1.0))
    for swc, report in runs:                # report: dict, JSON path or RunReport
        segment_id = exporter.add(swc, report)
    exporter.write("out/ng")
    upload_dir("out/ng", "s3://bucket/prefix")
    batch = exporter.batch_url("s3://bucket/prefix")
    one = exporter.neuron_url(segment_id, "s3://bucket/prefix")

``write`` produces a skeleton source (``skeletons/``) and one point-annotation
source per metric (``annotations/<metric>/``). Each vertex has ``n_flags`` (how
many metrics flagged it) and a ``flag_<metric>`` attribute for every metric
that flagged a node anywhere in the batch, and the default shader draws
flagged nodes in red. Each segment is tagged with the metrics that flagged it.

The batch link shows every neuron and the per-metric annotation layers. A
neuron's link instead carries its flagged nodes in the state, as one editable
``qc_errors`` layer (see :func:`review_layer`). Writing annotations needs the
``annotations`` extra; building states and links does not.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from neuron_ng.adapters.standard_morph_qc import (
    FLAG_STATUSES, STATUS_LABELS, FlaggedNode, load_report,
)
from neuron_ng.core.coords import CoordinateSpace
from neuron_ng.core.model import Neuron, Point, PropertySpec, Skeleton
from neuron_ng.io.swc import read_swc
from neuron_ng.precomputed.annotations import write_point_annotations
from neuron_ng.precomputed.skeletons import write_skeleton_source
from neuron_ng.state.layers import (
    AnnotationLayer, ImageLayer, LocalAnnotationLayer, SegmentationLayer,
)
from neuron_ng.state.shaders import ANNOTATION_COLORS, highlight_shader
from neuron_ng.state.viewer import DEFAULT_BASE_URL, ViewerState

SKELETON_DIR = "skeletons"
ANNOTATION_DIR = "annotations"
SKELETON_LAYER = "neurons"
IMAGE_LAYER = "image"
REVIEW_LAYER = "qc_errors"
#: Name of the annotation relationship that points at the skeleton segment.
RELATIONSHIP = "neuron"
N_FLAGS = "n_flags"

#: Draws nodes flagged by any metric in red, and everything else in the
#: segment's own color.
FLAG_SHADER = highlight_shader(N_FLAGS)

NODE_ID_PROPERTY = PropertySpec("node_id", "uint32")
STATUS_PROPERTY = PropertySpec.enum("status", STATUS_LABELS)


def flag_attribute(metric):
    """Name of the per-vertex attribute that marks nodes flagged by ``metric``."""
    return f"flag_{metric}"


def flag_count_property(metric):
    """Name of the segment property counting the nodes ``metric`` flagged."""
    return f"n_flagged_{metric}"


@dataclass
class NeuronRecord:
    segment_id: int
    label: str
    #: None when the morphology could not be built (a BUILD integrity failure).
    skeleton: Optional[Skeleton]
    #: metric name -> flagged nodes, in report order (integrity results first).
    flags: Dict[str, List[FlaggedNode]] = field(default_factory=dict)

    @property
    def position(self):
        """Where to center a view of this neuron: its first flagged node, else
        its soma, else its first node. None if nothing has coordinates."""
        for nodes in self.flags.values():
            if nodes:
                return nodes[0].xyz
        return self.skeleton.home_position() if self.skeleton is not None else None


def review_layer(flags, segment_id=None, name=REVIEW_LAYER):
    """Return a `LocalAnnotationLayer` with a point at every flagged node.

    Each point's description starts as ``"<metric> · <status> · node <id>"``,
    and its properties are the node id, status and metric. A node flagged
    twice by one metric is listed once.

    Parameters
    ----------
    flags : dict[str, list[FlaggedNode]]
        ``{metric: flagged nodes}`` for one neuron, as in ``NeuronRecord.flags``.
    segment_id : int, optional
        The neuron's skeleton segment. When given, each point is related to it
        and the layer is linked to the skeleton layer.
    """
    metrics = list(flags)
    metric_property = PropertySpec.enum("metric", metrics, type="uint16")
    points, seen = [], set()
    for m, metric in enumerate(metrics):
        for node in flags[metric]:
            key = (metric, node.node_id)
            if key in seen:
                continue
            seen.add(key)
            points.append(Point(
                xyz=node.xyz,
                id=f"{metric}-{node.node_id}",
                description=f"{metric} · {node.status} · node {node.node_id}",
                props={
                    "node_id": node.node_id,
                    "status": STATUS_PROPERTY.code(node.status),
                    "metric": m,
                },
                segments={RELATIONSHIP: segment_id} if segment_id is not None else {},
            ))
    return LocalAnnotationLayer(
        name=name,
        points=points,
        properties=[NODE_ID_PROPERTY, STATUS_PROPERTY, metric_property],
        link=(RELATIONSHIP, SKELETON_LAYER) if segment_id is not None else None,
    )


class QCReviewExporter:
    """Accumulate QC runs, then write their layers and build links.

    Parameters
    ----------
    scale_um : sequence of 3 floats
        Micrometers per SWC unit along x, y, z: ``(1, 1, 1)`` for SWCs in
        micrometers, the voxel size for SWCs in voxel indices. Give this or
        ``space``.
    flag_statuses : sequence of str
        Result statuses whose flagged nodes are shown.
    space : CoordinateSpace, optional
        Instead of ``scale_um``, e.g. to add an affine.
    """

    def __init__(self, scale_um=None, flag_statuses=FLAG_STATUSES, space=None):
        if (scale_um is None) == (space is None):
            raise ValueError("Give exactly one of scale_um and space")
        self.space = CoordinateSpace.coerce(space if space is not None else scale_um)
        self.flag_statuses = tuple(flag_statuses)
        self.records = []

    @property
    def scale_um(self):
        return self.space.scale_um

    def add(self, swc, report, label=None):
        """Record one QC run and return the neuron's segment id (1, 2, ...).

        ``swc`` is the SWC path the report is for (or a `Skeleton`), and
        ``report`` its QC report: a dict, a JSON path, or a ``RunReport``.
        When a BUILD integrity check failed (``morphology_evaluated`` is
        False) the neuron gets no skeleton, but its flagged nodes that have
        coordinates are still shown.
        """
        qc = load_report(report)
        segment_id = len(self.records) + 1
        skeleton = None
        if qc.morphology_evaluated:
            skeleton = swc if isinstance(swc, Skeleton) else read_swc(swc)
        flags = qc.flagged_nodes(
            self.flag_statuses, skeleton.xyz_by_id() if skeleton is not None else None,
        )
        self.records.append(NeuronRecord(
            segment_id=segment_id,
            label=label or _default_label(swc, qc.input_ref, segment_id),
            skeleton=skeleton,
            flags=flags,
        ))
        return segment_id

    @property
    def metrics(self):
        """Metrics that flagged at least one node, in first-seen order."""
        seen = {}
        for record in self.records:
            for metric in record.flags:
                seen.setdefault(metric, None)
        return list(seen)

    def record(self, segment_id):
        return self.records[segment_id - 1]

    # ------------------------------------------------------------------ write
    def neurons(self, metrics=None):
        """The records as `Neuron`s, with flag attributes, tags and counts."""
        metrics = self.metrics if metrics is None else metrics
        return [_to_neuron(record, metrics) for record in self.records]

    def write(self, out_dir):
        """Write ``skeletons/`` and ``annotations/<metric>/`` under ``out_dir``.

        Returns ``{"skeletons": [segment ids], "annotations": [metrics]}``.
        """
        out_dir = Path(out_dir)
        metrics = self.metrics
        segments = write_skeleton_source(
            self.neurons(metrics), out_dir / SKELETON_DIR, self.space,
            attribute_names=[N_FLAGS] + [flag_attribute(m) for m in metrics],
            tag_vocabulary=metrics,
            number_types={flag_count_property(m): "uint32" for m in metrics},
        )
        annotated = []
        for metric in metrics:
            points = [
                Point(
                    xyz=node.xyz,
                    props={"node_id": node.node_id, "status": STATUS_PROPERTY.code(node.status)},
                    segments={RELATIONSHIP: record.segment_id},
                )
                for record in self.records for node in record.flags.get(metric, ())
            ]
            if write_point_annotations(
                points, out_dir / ANNOTATION_DIR / metric, self.space,
                properties=[NODE_ID_PROPERTY, STATUS_PROPERTY], relationships=[RELATIONSHIP],
            ):
                annotated.append(metric)
        return {"skeletons": segments, "annotations": annotated}

    # ------------------------------------------------------------ states/links
    def sources(self, source_url, metrics=None):
        """Return ``(skeleton_source, {metric: annotation_source})`` for layers
        written by :meth:`write` and published at ``source_url``."""
        base = source_url.rstrip("/")
        metrics = self.metrics if metrics is None else metrics
        return (
            f"precomputed://{base}/{SKELETON_DIR}",
            {m: f"precomputed://{base}/{ANNOTATION_DIR}/{m}" for m in metrics},
        )

    def batch_view(self, source_url, image_source=None):
        """`ViewerState` showing every neuron and every metric's errors."""
        skeleton_source, annotation_sources = self.sources(source_url)
        segments = [r.segment_id for r in self.records if r.skeleton is not None]
        view = self._base_view(image_source)
        if segments:
            view.add(self._skeleton_layer(skeleton_source, segments))
        for i, (metric, source) in enumerate(annotation_sources.items()):
            view.add(AnnotationLayer(
                source=source, name=metric,
                color=ANNOTATION_COLORS[i % len(ANNOTATION_COLORS)],
                link=(RELATIONSHIP, SKELETON_LAYER) if segments else None,
            ))
        view.set_position(next((r.position for r in self.records if r.position is not None), None))
        if segments:
            view.select(SKELETON_LAYER)
        return view

    def neuron_view(self, segment_id, source_url, image_source=None):
        """`ViewerState` showing one neuron, centered on its first flagged node
        (or its soma), with its flagged nodes in one editable review layer."""
        record = self.record(segment_id)
        skeleton_source, _ = self.sources(source_url, metrics=[])
        has_skeleton = record.skeleton is not None
        view = self._base_view(image_source)
        if has_skeleton:
            view.add(self._skeleton_layer(skeleton_source, [segment_id]))
        if record.flags:
            view.add(review_layer(record.flags, segment_id if has_skeleton else None))
        view.set_position(record.position)
        if record.flags:
            view.select(REVIEW_LAYER)
        elif has_skeleton:
            view.select(SKELETON_LAYER)
        return view

    def batch_state(self, source_url, image_source=None):
        return self.batch_view(source_url, image_source).to_json()

    def neuron_state(self, segment_id, source_url, image_source=None):
        return self.neuron_view(segment_id, source_url, image_source).to_json()

    def batch_url(self, source_url, image_source=None, base_url=DEFAULT_BASE_URL):
        return self.batch_view(source_url, image_source).to_url(base_url)

    def neuron_url(self, segment_id, source_url, image_source=None, base_url=DEFAULT_BASE_URL):
        return self.neuron_view(segment_id, source_url, image_source).to_url(base_url)

    def _base_view(self, image_source):
        view = ViewerState(self.space)
        if image_source:
            view.add(ImageLayer(image_source, name=IMAGE_LAYER))
        return view

    @staticmethod
    def _skeleton_layer(source, segments):
        return SegmentationLayer(
            source=source, name=SKELETON_LAYER, segments=segments, skeleton_shader=FLAG_SHADER,
        )


def _to_neuron(record, metrics):
    neuron = Neuron(
        segment_id=record.segment_id,
        label=record.label,
        skeleton=record.skeleton,
        tags=list(record.flags),
        numbers={flag_count_property(m): len(record.flags.get(m, ())) for m in metrics},
    )
    skeleton = record.skeleton
    if skeleton is not None:
        index_of = skeleton.index_of()
        n_flags = np.zeros(len(skeleton), dtype="<f4")
        for metric in metrics:
            flag = np.zeros(len(skeleton), dtype="<f4")
            for node in record.flags.get(metric, ()):
                i = index_of.get(node.node_id)
                if i is not None:
                    flag[i] = 1.0
            n_flags += flag
            neuron.vertex_attributes[flag_attribute(metric)] = flag
        neuron.vertex_attributes[N_FLAGS] = n_flags
    return neuron


def _default_label(swc, input_ref, segment_id):
    path = input_ref or (swc if isinstance(swc, (str, os.PathLike)) else None)
    return Path(path).stem if path else f"neuron_{segment_id}"


__all__ = [
    "QCReviewExporter",
    "NeuronRecord",
    "review_layer",
    "flag_attribute",
    "flag_count_property",
    "FLAG_SHADER",
    "SKELETON_DIR",
    "ANNOTATION_DIR",
    "SKELETON_LAYER",
    "IMAGE_LAYER",
    "REVIEW_LAYER",
    "RELATIONSHIP",
]
