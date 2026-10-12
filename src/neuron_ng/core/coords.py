"""The coordinate space that layers and views are written in.

Every layer is written in the reconstruction's own units and never converted.
What changes is the *scale*, the physical size in micrometers of one unit along
x, y and z. For SWCs already in micrometers (including CCF-registered SWCs) the
scale is ``(1, 1, 1)``. For SWCs in voxel indices it is the voxel size (e.g.
``(0.748, 0.748, 1.0)`` for exaSPIM). Keeping the raw coordinates means node
positions from other tools (e.g. a QC report) are valid Neuroglancer positions
as they are.

An optional 3x4 ``affine`` maps the reconstruction's coordinates into the
viewer's (still in reconstruction units), for SWCs that are offset, flipped or
rotated relative to the image they are shown over. It is applied by Neuroglancer
through layer transforms, so the written data stays in reconstruction units.
"""
import numpy as np

AXES = ("x", "y", "z")
_IDENTITY = np.hstack([np.eye(3), np.zeros((3, 1))])


def validate_scale(scale_um):
    """Return ``scale_um`` as a float ``(3,)`` array, or raise ``ValueError``."""
    scale = np.asarray(scale_um, dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(
            f"scale_um must be three positive numbers (um per unit along x, y, z); "
            f"got {scale_um!r}"
        )
    return scale


def _validate_affine(affine):
    if affine is None:
        return _IDENTITY.copy()
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape == (4, 4):
        if not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError(f"A 4x4 affine must have a last row of [0, 0, 0, 1]; got {affine!r}")
        matrix = matrix[:3]
    if matrix.shape != (3, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"affine must be a finite 3x4 (or 4x4) matrix; got {affine!r}")
    return matrix


class CoordinateSpace:
    """Units of a set of reconstructions: ``scale_um`` and an optional ``affine``.

    Parameters
    ----------
    scale_um : sequence of 3 floats
        Micrometers per unit along x, y, z.
    affine : array-like (3, 4) or (4, 4), optional
        Maps reconstruction coordinates to viewer coordinates, both in
        reconstruction units. Identity when omitted.
    """

    def __init__(self, scale_um, affine=None):
        self.scale_um = tuple(float(v) for v in validate_scale(scale_um))
        self.affine = _validate_affine(affine)

    def __repr__(self):
        if self.is_identity:
            return f"CoordinateSpace(scale_um={self.scale_um})"
        return f"CoordinateSpace(scale_um={self.scale_um}, affine={self.affine.tolist()})"

    def __eq__(self, other):
        return (
            isinstance(other, CoordinateSpace)
            and self.scale_um == other.scale_um
            and np.array_equal(self.affine, other.affine)
        )

    @classmethod
    def coerce(cls, value):
        """Return ``value`` if it is a ``CoordinateSpace``, else ``cls(value)``
        (a bare scale)."""
        return value if isinstance(value, cls) else cls(value)

    @property
    def is_identity(self):
        return np.array_equal(self.affine, _IDENTITY)

    def dimensions_json(self):
        """Neuroglancer ``dimensions`` JSON: ``{"x": [scale_m, "m"], ...}``."""
        return {axis: [s * 1e-6, "m"] for axis, s in zip(AXES, self.scale_um)}

    def apply(self, xyz):
        """Map one reconstruction point to viewer coordinates (a float tuple)."""
        point = np.asarray(xyz, dtype=float)
        if not self.is_identity:
            point = self.affine[:, :3] @ point + self.affine[:, 3]
        return tuple(float(v) for v in point)

    def source_transform(self):
        """``transform`` JSON for a layer source holding reconstruction-unit
        data: ``outputDimensions``, plus the affine's ``matrix`` unless it is
        the identity."""
        transform = {}
        if not self.is_identity:
            transform["matrix"] = self.affine.tolist()
        transform["outputDimensions"] = self.dimensions_json()
        return transform

    def skeleton_transform_nm(self):
        """Row-major 3x4 transform from reconstruction units to nanometers, for
        a precomputed skeleton ``info``.

        Standalone precomputed skeletons are defined in nanometers, so this is
        ``diag(scale_nm) @ affine``.
        """
        scale_nm = np.asarray(self.scale_um) * 1000.0
        transform = scale_nm[:, None] * self.affine
        return [float(v) for v in transform.ravel()]

    def coordinate_space(self):
        """Return a ``neuroglancer.CoordinateSpace`` for writing precomputed
        annotations. Requires the ``annotations`` extra."""
        from neuron_ng._deps import import_neuroglancer

        ng = import_neuroglancer()
        return ng.CoordinateSpace(json=self.dimensions_json())
