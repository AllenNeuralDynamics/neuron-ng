"""Layer specs for viewer states.

Each layer is a small dataclass whose ``to_json(space)`` returns the layer's
JSON. ``space`` is the `CoordinateSpace` of the reconstructions, used by layers
that hold or reference reconstruction-unit data. No ``neuroglancer`` package is
needed.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from neuron_ng.core.model import Point, PropertySpec
from neuron_ng.state.shaders import (
    ANNOTATION_COLORS, DEFAULT_ANNOTATION_SHADER, DEFAULT_SKELETON_SHADER,
)

LOCAL_ANNOTATIONS = "local://annotations"


@dataclass
class ImageLayer:
    """Any Neuroglancer image source (e.g. ``zarr://s3://.../fused.zarr``, a
    CCF template), shown as is."""
    source: str
    name: str = "image"
    shader: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def to_json(self, space=None):
        layer = {"type": "image", "name": self.name, "source": self.source}
        if self.shader:
            layer["shader"] = self.shader
        layer.update(self.extra)
        return layer


@dataclass
class SegmentationLayer:
    """A segmentation source: a precomputed skeleton source written by this
    package, or e.g. a CCF annotation volume.

    ``skeleton_shader`` adds ``skeletonRendering`` (skeleton sources only).
    """
    source: str
    name: str = "neurons"
    segments: Sequence[int] = ()
    skeleton_shader: Optional[str] = DEFAULT_SKELETON_SHADER
    mode2d: str = "lines_and_points"
    mode3d: str = "lines"
    extra: Dict[str, object] = field(default_factory=dict)

    def to_json(self, space=None):
        layer = {
            "type": "segmentation",
            "name": self.name,
            "source": self.source,
            "segments": [str(s) for s in self.segments],
        }
        if self.skeleton_shader:
            layer["skeletonRendering"] = {
                "shader": self.skeleton_shader, "mode2d": self.mode2d, "mode3d": self.mode3d,
            }
        layer.update(self.extra)
        return layer


@dataclass
class AnnotationLayer:
    """A precomputed annotation source written in reconstruction units.

    ``link`` is ``(relationship, segmentation layer name)``. With
    ``filter_by_segmentation`` only points related to the selected segments
    are shown.
    """
    source: str
    name: str
    color: str = ANNOTATION_COLORS[0]
    shader: str = DEFAULT_ANNOTATION_SHADER
    link: Optional[Tuple[str, str]] = None
    filter_by_segmentation: bool = True

    def to_json(self, space):
        source = self.source
        if not space.is_identity:
            source = {"url": self.source, "transform": space.source_transform()}
        layer = {
            "type": "annotation",
            "name": self.name,
            "source": source,
            "tab": "annotations",
            "annotationColor": self.color,
            "shader": self.shader,
        }
        if self.link:
            relationship, segmentation = self.link
            layer["linkedSegmentationLayer"] = {relationship: segmentation}
            if self.filter_by_segmentation:
                layer["filterBySegmentation"] = [relationship]
        return layer


@dataclass
class LocalAnnotationLayer:
    """Point annotations stored in the state itself.

    Unlike a precomputed source, Neuroglancer lists local annotations in the
    layer's Annotations tab (click one to jump to it) and lets the viewer edit
    each description or delete points. Edits are kept in the link.

    Each `Point` needs an ``id``, and ``props`` with a value for every spec in
    ``properties`` (enum properties take the stored code). ``segments`` is
    keyed by relationship, and ``link`` is ``(relationship, segmentation layer
    name)``.
    """
    name: str
    points: List[Point] = field(default_factory=list)
    properties: Sequence[PropertySpec] = ()
    color: str = ANNOTATION_COLORS[0]
    shader: str = DEFAULT_ANNOTATION_SHADER
    link: Optional[Tuple[str, str]] = None

    def to_json(self, space):
        relationship = self.link[0] if self.link else None
        annotations = []
        for point in self.points:
            annotation = {"type": "point", "id": point.id, "point": [float(v) for v in point.xyz]}
            if point.description is not None:
                annotation["description"] = point.description
            annotation["props"] = [point.props[spec.id] for spec in self.properties]
            if relationship is not None and relationship in point.segments:
                annotation["segments"] = [[str(s) for s in _as_list(point.segments[relationship])]]
            annotations.append(annotation)

        layer = {
            "type": "annotation",
            "name": self.name,
            "source": {"url": LOCAL_ANNOTATIONS, "transform": space.source_transform()},
            "tab": "annotations",
            "annotationColor": self.color,
            "shader": self.shader,
            "annotationProperties": [spec.to_json() for spec in self.properties],
            "annotations": annotations,
        }
        if self.link:
            relationship, segmentation = self.link
            layer["annotationRelationships"] = [relationship]
            layer["linkedSegmentationLayer"] = {relationship: segmentation}
        return layer


def _as_list(ids):
    return [ids] if isinstance(ids, int) else list(ids)
