"""Shared test helpers."""
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import neuroglancer
except ImportError:  # the annotations extra needs Python >= 3.10
    neuroglancer = None

needs_neuroglancer = unittest.skipIf(neuroglancer is None, "neuroglancer extra not installed")

# Soma, then an axon whose 3 -> 4 edge is 80 units long.
LONG_EDGE_SWC = (
    "1 1 0 0 0 1 -1\n"
    "2 2 10 0 0 1 1\n"
    "3 2 20 0 0 1 2\n"
    "4 2 100 0 0 1 3\n"
    "5 2 110 0 0 1 4\n"
)
CLEAN_SWC = "1 1 5 6 7 1 -1\n2 2 10 6 7 1 1\n"


def result(name, status="pass", ids=(), coords=()):
    """One result as it appears in a report's JSON."""
    return {
        "name": name, "status": status, "message": "", "value": None,
        "flagged_node_ids": list(ids), "flagged_node_coordinates": [list(c) for c in coords],
    }


def report(results=(), integrity=(), morphology_evaluated=True, input_ref=None,
           schema_version="1.2"):
    """A minimal ``RunReport.to_dict()``-shaped report."""
    return {
        "schema_version": schema_version, "standard_morph_version": "1.0.1",
        "input_ref": input_ref, "morphology_evaluated": morphology_evaluated,
        "integrity_results": list(integrity), "results": list(results),
        "summary": {},
    }


def long_edge_report(**kwargs):
    """``edge_length`` flags node 4 of LONG_EDGE_SWC."""
    return report([result("edge_length", "fail", [4], [(100.0, 0.0, 0.0)])], **kwargs)


def decode_skeleton(data, attribute_names):
    n_vertices, n_edges = struct.unpack_from("<II", data)
    offset = 8
    vertices = np.frombuffer(data, "<f4", n_vertices * 3, offset).reshape(-1, 3)
    offset += vertices.nbytes
    edges = np.frombuffer(data, "<u4", n_edges * 2, offset).reshape(-1, 2)
    offset += edges.nbytes
    attributes = {}
    for name in attribute_names:
        attributes[name] = np.frombuffer(data, "<f4", n_vertices, offset)
        offset += n_vertices * 4
    assert offset == len(data)
    return vertices, edges, attributes


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def swc(self, name, text):
        path = self.tmp / name
        path.write_text(text)
        return path
