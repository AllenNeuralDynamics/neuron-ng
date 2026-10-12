"""Write points as a Neuroglancer precomputed point-annotation source.

Each point carries typed properties and may be related to skeleton segments,
which lets an annotation layer linked to a skeleton layer show only the
selected neurons' points.

Requires the ``annotations`` extra (``neuroglancer.write_annotations``).
"""
from pathlib import Path

from neuron_ng._deps import import_neuroglancer
from neuron_ng.core.coords import CoordinateSpace


def write_point_annotations(points, out_dir, space, properties=(), relationships=()):
    """Write ``points`` to ``out_dir``, unless there are none.

    Parameters
    ----------
    points : iterable of Point
        ``Point.props`` holds a value for every spec in ``properties`` (enum
        properties take the stored code, see ``PropertySpec.code``), and
        ``Point.segments`` the segment id(s) for each relationship.
    out_dir : str | Path
    space : CoordinateSpace | sequence of 3 floats
        Units of the points (a bare scale is micrometers per unit).
    properties : sequence of PropertySpec
    relationships : sequence of str

    Returns True if a source was written.
    """
    ng = import_neuroglancer()
    from neuroglancer.write_annotations import AnnotationWriter

    specs = []
    for spec in properties:
        kwargs = {"id": spec.id, "type": spec.type}
        if spec.description:
            kwargs["description"] = spec.description
        if spec.enum_labels is not None:
            kwargs["enum_values"] = list(range(len(spec.enum_labels)))
            kwargs["enum_labels"] = list(spec.enum_labels)
        specs.append(ng.AnnotationPropertySpec(**kwargs))

    writer = AnnotationWriter(
        coordinate_space=CoordinateSpace.coerce(space).coordinate_space(),
        annotation_type="point", relationships=list(relationships), properties=specs,
    )
    for point in points:
        writer.add_point(point.xyz, **point.props, **point.segments)
    if not writer.annotations:
        return False
    writer.write(Path(out_dir))
    return True
