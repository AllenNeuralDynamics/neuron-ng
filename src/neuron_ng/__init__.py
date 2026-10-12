"""Neuroglancer layers and views for neuron reconstructions.

The package is layered so that each part can be used on its own:

* ``neuron_ng.core``: the coordinate space and a neutral data model
  (`Skeleton`, `Neuron`, `Point`, `PropertySpec`).
* ``neuron_ng.io``: readers that build the model (``io.swc``).
* ``neuron_ng.precomputed``: writers for precomputed skeleton, segment-property
  and point-annotation sources.
* ``neuron_ng.state``: layer specs, shaders and the `ViewerState` builder, as
  plain JSON (links never need the ``neuroglancer`` package).
* ``neuron_ng.adapters``: readers for other tools' outputs, e.g.
  ``adapters.standard_morph_qc`` for standard_morph QC reports.
* ``neuron_ng.recipes``: ready-made views built from the above, e.g.
  ``recipes.qc_review`` and ``recipes.gallery``.
* ``neuron_ng.storage``: upload written layers to S3.
"""
from neuron_ng.core import CoordinateSpace, Neuron, Point, PropertySpec, Skeleton
from neuron_ng.io.swc import read_swc
from neuron_ng.recipes.gallery import GalleryExporter
from neuron_ng.recipes.qc_review import QCReviewExporter
from neuron_ng.state import ViewerState, state_to_url

try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("neuron-ng")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "CoordinateSpace",
    "Neuron",
    "Point",
    "PropertySpec",
    "Skeleton",
    "read_swc",
    "GalleryExporter",
    "QCReviewExporter",
    "ViewerState",
    "state_to_url",
]
