import json
import struct
import types
import unittest

import numpy as np

from neuron_ng.core.model import Neuron, Point, PropertySpec
from neuron_ng.io.swc import read_swc
from neuron_ng.precomputed import (
    encode_skeleton, segment_properties_info, write_point_annotations, write_skeleton_source,
)
from tests.helpers import LONG_EDGE_SWC, TmpCase, decode_skeleton, needs_neuroglancer

SCALE = (0.5, 0.5, 2.0)


class TestSkeletonSource(TmpCase):
    def test_writes_info_skeletons_and_properties(self):
        skeleton = read_swc(self.swc("a.swc", LONG_EDGE_SWC))
        neurons = [
            Neuron(1, "a", skeleton, vertex_attributes={"w": np.arange(5)}, tags=["x"],
                   numbers={"count": 3}),
            Neuron(2, "unbuilt"),
        ]
        out = self.tmp / "skeletons"
        written = write_skeleton_source(neurons, out, SCALE, attribute_names=["w"],
                                        tag_vocabulary=["y", "x"], number_types={"count": "uint32"})
        self.assertEqual(written, [1])
        self.assertFalse((out / "2").exists())

        info = json.loads((out / "info").read_text())
        self.assertEqual(info["@type"], "neuroglancer_skeletons")
        self.assertEqual(info["transform"][0], 500.0)
        self.assertEqual([a["id"] for a in info["vertex_attributes"]], ["radius", "w"])
        self.assertEqual(info["segment_properties"], "segment_properties")

        vertices, edges, attributes = decode_skeleton((out / "1").read_bytes(), ["radius", "w"])
        np.testing.assert_array_equal(vertices[:, 0], [0, 10, 20, 100, 110])
        self.assertEqual(edges.tolist(), [[1, 0], [2, 1], [3, 2], [4, 3]])
        np.testing.assert_array_equal(attributes["w"], [0, 1, 2, 3, 4])

        props = json.loads((out / "segment_properties" / "info").read_text())["inline"]
        self.assertEqual(props["ids"], ["1", "2"])
        label, tags, count = props["properties"]
        self.assertEqual(label["values"], ["a", "unbuilt"])
        self.assertEqual((tags["tags"], tags["values"]), (["y", "x"], [[1], []]))
        self.assertEqual((count["data_type"], count["values"]), ("uint32", [3, 0]))

    def test_missing_attribute_rejected(self):
        skeleton = read_swc(self.swc("a.swc", LONG_EDGE_SWC))
        with self.assertRaises(ValueError):
            write_skeleton_source([Neuron(1, "a", skeleton)], self.tmp / "s", SCALE, ["w"])

    def test_encoding_layout(self):
        data = encode_skeleton([[1, 2, 3], [4, 5, 6]], [[1, 0]], {"radius": [7, 8]})
        self.assertEqual(struct.unpack_from("<II", data), (2, 1))
        self.assertEqual(len(data), 8 + 24 + 8 + 8)

    @needs_neuroglancer
    def test_matches_neuroglancer_encoding(self):
        import neuroglancer.skeleton

        xyz = np.array([[1, 2, 3], [4, 5, 6]], dtype="<f4")
        edges = np.array([[1, 0]], dtype="<u4")
        radius = np.array([7, 8], dtype="<f4")
        # encode() reads only the source's vertex_attributes.
        source = types.SimpleNamespace(vertex_attributes={
            "radius": neuroglancer.skeleton.VertexAttributeInfo(np.float32, 1),
        })
        expected = neuroglancer.skeleton.Skeleton(xyz, edges, vertex_attributes={"radius": radius}) \
            .encode(source)
        self.assertEqual(encode_skeleton(xyz, edges, {"radius": radius}), expected)

    def test_segment_properties_ignore_unknown_tags(self):
        info = segment_properties_info([Neuron(1, "a", tags=["b", "zzz"])], tag_vocabulary=["a", "b"])
        self.assertEqual(info["inline"]["properties"][1]["values"], [[1]])


@needs_neuroglancer
class TestPointAnnotations(TmpCase):
    def test_points_with_properties_and_relationship(self):
        status = PropertySpec.enum("status", ["fail", "review"])
        points = [Point((100.0, 0.0, 0.0), props={"node_id": 4, "status": 1}, segments={"neuron": 7})]
        out = self.tmp / "ann"
        self.assertTrue(write_point_annotations(
            points, out, SCALE, properties=[PropertySpec("node_id", "uint32"), status],
            relationships=["neuron"],
        ))
        info = json.loads((out / "info").read_text())
        self.assertEqual(info["annotation_type"], "point")
        self.assertEqual(info["dimensions"]["z"], [2e-06, "m"])
        self.assertEqual([p["id"] for p in info["properties"]], ["node_id", "status"])
        self.assertEqual(info["properties"][1]["enum_labels"], ["fail", "review"])
        self.assertEqual(info["relationships"], [{"id": "neuron", "key": "rel_neuron"}])
        data = (out / "by_id" / "0").read_bytes()
        self.assertEqual(struct.unpack_from("<3f", data), (100.0, 0.0, 0.0))
        self.assertEqual(struct.unpack_from("<I", data, 12)[0], 4)
        self.assertEqual(struct.unpack_from("<Q", data, len(data) - 8)[0], 7)
        self.assertTrue((out / "rel_neuron" / "7").exists())

    def test_no_points_writes_nothing(self):
        self.assertFalse(write_point_annotations([], self.tmp / "ann", SCALE))
        self.assertFalse((self.tmp / "ann").exists())


if __name__ == "__main__":
    unittest.main()
