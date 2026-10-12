"""Neutral data model shared by the writers, states and recipes.

Nothing here knows where the data came from (SWC, QC report, proofreading
tool) or what a view is for. Adapters build these objects, and recipes decide
what to show.

* `Skeleton`: one reconstruction's nodes and edges, keyed by original node id.
* `Neuron`: a skeleton (or None) as one segment, with a label, tags and numeric
  properties for the segment-properties table.
* `PropertySpec` / `Point`: typed point annotations, for precomputed or local
  annotation layers.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: SWC compartment code for the soma.
SOMA_COMPARTMENT = 1


@dataclass
class Skeleton:
    node_id: np.ndarray  # (N,) original node ids
    xyz: np.ndarray      # (N, 3) coordinates, in the reconstruction's units
    radius: np.ndarray   # (N,)
    edges: np.ndarray    # (E, 2) vertex index pairs (child, parent)
    soma_index: Optional[int] = None
    compartment: Optional[np.ndarray] = None  # (N,) SWC type codes, if known

    @classmethod
    def from_swc_arrays(cls, node_id, compartment, xyz, radius, parent_id):
        """Build from SWC columns. Node ids need not be contiguous or sorted.

        A parent id of -1 marks a root. A parent id absent from the file gives
        the node no edge. The soma is the first node that is both a soma
        compartment and a root (a stray soma-typed node mid-tree is not).
        """
        node_id = np.asarray(node_id, dtype=np.int64)
        parent_id = np.asarray(parent_id, dtype=np.int64)
        compartment = np.asarray(compartment, dtype=np.int64)
        index_of = {int(nid): i for i, nid in enumerate(node_id)}
        parent = np.array(
            [-1 if p == -1 else index_of.get(int(p), -2) for p in parent_id], dtype=np.int64,
        )
        child = np.flatnonzero(parent >= 0)
        edges = np.stack([child, parent[child]], axis=1) if child.size else np.zeros((0, 2))
        soma = np.flatnonzero((compartment == SOMA_COMPARTMENT) & (parent == -1))
        return cls(
            node_id=node_id,
            xyz=np.asarray(xyz, dtype=float).reshape(-1, 3),
            radius=np.asarray(radius, dtype=float),
            edges=np.asarray(edges, dtype=np.int64),
            soma_index=int(soma[0]) if soma.size else None,
            compartment=compartment,
        )

    def __len__(self):
        return len(self.node_id)

    def index_of(self):
        """``{original node id: vertex index}``."""
        return {int(nid): i for i, nid in enumerate(self.node_id)}

    def xyz_by_id(self):
        """``{original node id: (3,) coordinates}``."""
        return {int(nid): self.xyz[i] for i, nid in enumerate(self.node_id)}

    def home_position(self):
        """The soma, else the first node, as a float tuple. None if empty."""
        if not len(self):
            return None
        index = self.soma_index if self.soma_index is not None else 0
        return tuple(float(v) for v in self.xyz[index])


@dataclass
class Neuron:
    """One segment of a skeleton source.

    ``skeleton`` is None for a neuron that could not be drawn (e.g. a file that
    failed to parse). It still gets a row in the segment-properties table.
    """
    segment_id: int
    label: str
    skeleton: Optional[Skeleton] = None
    #: ``{name: (N,) array}`` per-vertex attributes (besides ``radius``).
    vertex_attributes: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Names from the source's tag vocabulary that apply to this neuron.
    tags: List[str] = field(default_factory=list)
    #: ``{name: number}`` numeric segment properties.
    numbers: Dict[str, float] = field(default_factory=dict)


@dataclass
class PropertySpec:
    """A typed annotation property (``uint8``, ``uint16``, ``uint32``,
    ``int*``, ``float32`` or ``rgb``/``rgba``), optionally an enum."""
    id: str
    type: str
    enum_labels: Optional[Sequence[str]] = None
    description: Optional[str] = None

    @classmethod
    def enum(cls, id, labels, type="uint8", description=None):
        return cls(id=id, type=type, enum_labels=list(labels), description=description)

    def code(self, label):
        """The stored value of an enum ``label``."""
        return list(self.enum_labels).index(label)

    def to_json(self):
        spec = {"id": self.id, "type": self.type}
        if self.description:
            spec["description"] = self.description
        if self.enum_labels is not None:
            spec["enum_values"] = list(range(len(self.enum_labels)))
            spec["enum_labels"] = list(self.enum_labels)
        return spec


@dataclass
class Point:
    """One point annotation. ``props`` is in the order of the layer's
    ``PropertySpec``s, and ``segments`` maps a relationship to segment ids."""
    xyz: Tuple[float, float, float]
    props: Dict[str, object] = field(default_factory=dict)
    segments: Dict[str, Sequence[int]] = field(default_factory=dict)
    id: Optional[str] = None
    description: Optional[str] = None
