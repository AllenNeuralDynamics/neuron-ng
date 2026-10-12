"""Gallery views: many reconstructions in one skeleton source, with no QC.

Typical use::

    from neuron_ng.recipes.gallery import GalleryExporter
    from neuron_ng.state import SegmentationLayer

    gallery = GalleryExporter(scale_um=(1, 1, 1))     # e.g. SWCs in CCF micrometers
    for path in swc_paths:
        gallery.add(path, tags=["cortex"], numbers={"soma_depth_um": 412.0})
    gallery.write("out/gallery")                      # then upload to s3://bucket/gallery
    url = gallery.url(
        "s3://bucket/gallery",
        extra_layers=[SegmentationLayer("precomputed://s3://bucket/ccf/annotation_10",
                                        name="ccf", skeleton_shader=None)],
    )

Each vertex has ``radius`` and ``compartment`` (the SWC type), and the default
shader colors by compartment. Each segment has a label, its tags and the given
numbers (``float32``) plus ``n_nodes``, so Neuroglancer can search and filter
the neurons in the segmentation layer's Seg. tab.
"""
from pathlib import Path

import numpy as np

from neuron_ng.core.coords import CoordinateSpace
from neuron_ng.core.model import Neuron, Skeleton
from neuron_ng.io.swc import read_swc, swc_label
from neuron_ng.precomputed.skeletons import write_skeleton_source
from neuron_ng.state.layers import ImageLayer, SegmentationLayer
from neuron_ng.state.shaders import compartment_shader
from neuron_ng.state.viewer import DEFAULT_BASE_URL, ViewerState

SKELETON_DIR = "skeletons"
SKELETON_LAYER = "neurons"
COMPARTMENT = "compartment"
N_NODES = "n_nodes"


class GalleryExporter:
    """Accumulate reconstructions, then write one skeleton source and build links.

    Parameters
    ----------
    scale_um : sequence of 3 floats
        Micrometers per SWC unit along x, y, z. Give this or ``space``.
    space : CoordinateSpace, optional
        Instead of ``scale_um``, e.g. to add an affine.
    shader : str, optional
        Skeleton shader for the views (default: color by compartment).
    """

    def __init__(self, scale_um=None, space=None, shader=None):
        if (scale_um is None) == (space is None):
            raise ValueError("Give exactly one of scale_um and space")
        self.space = CoordinateSpace.coerce(space if space is not None else scale_um)
        self.shader = shader or compartment_shader(COMPARTMENT)
        self.neurons = []

    def add(self, swc, label=None, tags=(), numbers=None):
        """Add one reconstruction (an SWC path or a `Skeleton`) and return its
        segment id (1, 2, ...)."""
        skeleton = swc if isinstance(swc, Skeleton) else read_swc(swc)
        segment_id = len(self.neurons) + 1
        if label is None:
            label = f"neuron_{segment_id}" if isinstance(swc, Skeleton) else swc_label(swc)
        compartment = (
            skeleton.compartment if skeleton.compartment is not None
            else np.zeros(len(skeleton))
        )
        self.neurons.append(Neuron(
            segment_id=segment_id,
            label=label,
            skeleton=skeleton,
            vertex_attributes={COMPARTMENT: np.asarray(compartment, dtype="<f4")},
            tags=list(tags),
            numbers={N_NODES: len(skeleton), **(numbers or {})},
        ))
        return segment_id

    @property
    def tags(self):
        """Every tag, in first-seen order."""
        seen = {}
        for neuron in self.neurons:
            for tag in neuron.tags:
                seen.setdefault(tag, None)
        return list(seen)

    def number_types(self):
        types = {N_NODES: "uint32"}
        for neuron in self.neurons:
            for name in neuron.numbers:
                types.setdefault(name, "float32")
        return types

    def write(self, out_dir):
        """Write ``skeletons/`` under ``out_dir``. Returns the segment ids written."""
        return write_skeleton_source(
            self.neurons, Path(out_dir) / SKELETON_DIR, self.space,
            attribute_names=[COMPARTMENT], tag_vocabulary=self.tags,
            number_types=self.number_types(),
        )

    def view(self, source_url, image_source=None, segments=None, extra_layers=()):
        """`ViewerState` of the gallery published at ``source_url``.

        ``segments`` limits the visible neurons (default: all). ``extra_layers``
        are added on top, e.g. a CCF annotation `SegmentationLayer`.
        """
        segments = [n.segment_id for n in self.neurons] if segments is None else list(segments)
        view = ViewerState(self.space)
        if image_source:
            view.add(ImageLayer(image_source))
        view.add(SegmentationLayer(
            source=f"precomputed://{source_url.rstrip('/')}/{SKELETON_DIR}",
            name=SKELETON_LAYER, segments=segments, skeleton_shader=self.shader,
        ))
        for layer in extra_layers:
            view.add(layer)
        first = next((n for n in self.neurons if n.segment_id in set(segments)), None)
        if first is not None:
            view.set_position(first.skeleton.home_position())
        return view.select(SKELETON_LAYER)

    def url(self, source_url, image_source=None, segments=None, extra_layers=(),
            base_url=DEFAULT_BASE_URL):
        return self.view(source_url, image_source, segments, extra_layers).to_url(base_url)


__all__ = ["GalleryExporter", "SKELETON_DIR", "SKELETON_LAYER", "COMPARTMENT", "N_NODES"]
