"""Write neurons as a standalone Neuroglancer precomputed skeleton source.

Layout under ``out_dir``::

    info                        neuroglancer_skeletons metadata
    <segment_id>                one encoded skeleton per neuron
    segment_properties/info     labels, tags and numbers (see segment_properties)

Every vertex carries ``radius`` plus the attributes named in
``attribute_names``, which each neuron supplies in ``Neuron.vertex_attributes``.
The binary encoding matches ``neuroglancer.skeleton.Skeleton.encode``; this
module does not need the ``neuroglancer`` package.
"""
import json
import struct
from pathlib import Path

import numpy as np

from neuron_ng.core.coords import CoordinateSpace
from neuron_ng.precomputed.segment_properties import write_segment_properties

SEGMENT_PROPERTIES_DIR = "segment_properties"
RADIUS = "radius"


def encode_skeleton(xyz, edges, attributes):
    """Encode one skeleton in the precomputed format.

    ``attributes`` is an ordered ``{name: (N,) array}``; its order must match
    the ``vertex_attributes`` list in the source ``info``.
    """
    xyz = np.asarray(xyz, dtype="<f4")
    edges = np.asarray(edges, dtype="<u4").reshape(-1, 2)
    parts = [struct.pack("<II", xyz.shape[0], edges.shape[0]), xyz.tobytes(), edges.tobytes()]
    parts += [np.asarray(values, dtype="<f4").tobytes() for values in attributes.values()]
    return b"".join(parts)


def skeleton_info(space, attribute_names, segment_properties=SEGMENT_PROPERTIES_DIR):
    """The source ``info``: transform to nm, and one float32 attribute per name."""
    info = {
        "@type": "neuroglancer_skeletons",
        "transform": CoordinateSpace.coerce(space).skeleton_transform_nm(),
        "vertex_attributes": [
            {"id": name, "data_type": "float32", "num_components": 1}
            for name in [RADIUS] + list(attribute_names)
        ],
    }
    if segment_properties:
        info["segment_properties"] = segment_properties
    return info


def write_skeleton_source(neurons, out_dir, space, attribute_names=(), tag_vocabulary=(),
                          number_types=None):
    """Write every neuron that has a skeleton to ``out_dir``.

    Parameters
    ----------
    neurons : list[Neuron]
        Every neuron gets a segment-properties row; only those with a skeleton
        get a skeleton file.
    out_dir : str | Path
    space : CoordinateSpace | sequence of 3 floats
        The neurons' units (a bare scale is micrometers per unit).
    attribute_names : sequence of str
        Per-vertex attributes after ``radius``. Every neuron with a skeleton
        must have each one in ``vertex_attributes``.
    tag_vocabulary, number_types
        Passed to :func:`write_segment_properties`.

    Returns the segment ids written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attribute_names = list(attribute_names)
    info = skeleton_info(space, attribute_names)
    (out_dir / "info").write_text(json.dumps(info))

    written = []
    for neuron in neurons:
        skeleton = neuron.skeleton
        if skeleton is None:
            continue
        missing = [name for name in attribute_names if name not in neuron.vertex_attributes]
        if missing:
            raise ValueError(f"Neuron {neuron.segment_id} is missing vertex attributes {missing}")
        attributes = {RADIUS: skeleton.radius}
        attributes.update((name, neuron.vertex_attributes[name]) for name in attribute_names)
        data = encode_skeleton(skeleton.xyz, skeleton.edges, attributes)
        (out_dir / str(neuron.segment_id)).write_bytes(data)
        written.append(neuron.segment_id)

    write_segment_properties(
        out_dir / SEGMENT_PROPERTIES_DIR, neurons,
        tag_vocabulary=tag_vocabulary, number_types=number_types,
    )
    return written
