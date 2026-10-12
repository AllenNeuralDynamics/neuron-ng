import json
import unittest

import numpy as np

from neuron_ng.core.coords import CoordinateSpace
from neuron_ng.recipes.gallery import COMPARTMENT, SKELETON_DIR, GalleryExporter
from neuron_ng.state import SegmentationLayer
from tests.helpers import CLEAN_SWC, LONG_EDGE_SWC, TmpCase, decode_skeleton, needs_neuroglancer, neuroglancer


class TestGallery(TmpCase):
    def setUp(self):
        super().setUp()
        self.gallery = GalleryExporter(scale_um=(1, 1, 1))
        self.gallery.add(self.swc("a.swc", LONG_EDGE_SWC), tags=["cortex"], numbers={"depth": 1.5})
        self.gallery.add(self.swc("b.swc", CLEAN_SWC), tags=["thalamus", "cortex"])

    def test_write(self):
        self.assertEqual(self.gallery.write(self.tmp / "g"), [1, 2])
        out = self.tmp / "g" / SKELETON_DIR
        names = [a["id"] for a in json.loads((out / "info").read_text())["vertex_attributes"]]
        self.assertEqual(names, ["radius", COMPARTMENT])
        _, _, attributes = decode_skeleton((out / "1").read_bytes(), names)
        np.testing.assert_array_equal(attributes[COMPARTMENT], [1, 2, 2, 2, 2])

        props = json.loads((out / "segment_properties" / "info").read_text())["inline"]["properties"]
        by_id = {p["id"]: p for p in props}
        self.assertEqual(by_id["label"]["values"], ["a", "b"])
        self.assertEqual(by_id["tags"]["tags"], ["cortex", "thalamus"])
        self.assertEqual(by_id["tags"]["values"], [[0], [0, 1]])
        self.assertEqual(by_id["n_nodes"]["values"], [5, 2])
        self.assertEqual((by_id["depth"]["data_type"], by_id["depth"]["values"]), ("float32", [1.5, 0]))

    def test_view_with_atlas_layer(self):
        atlas = SegmentationLayer("precomputed://s3://bucket/ccf", name="ccf", skeleton_shader=None)
        state = self.gallery.view("s3://bucket/g", image_source="zarr://img",
                                  segments=[2], extra_layers=[atlas]).to_json()
        self.assertEqual([l["name"] for l in state["layers"]], ["image", "neurons", "ccf"])
        self.assertEqual(state["layers"][1]["segments"], ["2"])
        self.assertIn("compartment", state["layers"][1]["skeletonRendering"]["shader"])
        self.assertEqual(state["position"], [5.0, 6.0, 7.0])  # neuron 2's soma

    def test_affine_space(self):
        space = CoordinateSpace((10, 10, 10), affine=[[-1, 0, 0, 100], [0, 1, 0, 0], [0, 0, 1, 0]])
        gallery = GalleryExporter(space=space)
        gallery.add(self.swc("a.swc", LONG_EDGE_SWC))
        self.assertEqual(gallery.view("s3://b/g").to_json()["position"], [100.0, 0.0, 0.0])
        gallery.write(self.tmp / "g")
        info = json.loads((self.tmp / "g" / SKELETON_DIR / "info").read_text())
        self.assertEqual(info["transform"][:4], [-1e4, 0.0, 0.0, 1e6])

    @needs_neuroglancer
    def test_url_round_trips(self):
        parsed = neuroglancer.parse_url(self.gallery.url("s3://bucket/g"))
        self.assertEqual(list(parsed.layers["neurons"].segments), [1, 2])


if __name__ == "__main__":
    unittest.main()
