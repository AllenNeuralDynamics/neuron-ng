import json
import unittest
import urllib.parse

import numpy as np

from neuron_ng.core.coords import CoordinateSpace
from neuron_ng.core.model import Point, PropertySpec
from neuron_ng.state import (
    AnnotationLayer, ImageLayer, LocalAnnotationLayer, SegmentationLayer, ViewerState,
    compartment_shader, highlight_shader, state_to_url,
)
from tests.helpers import needs_neuroglancer, neuroglancer

SCALE = (0.5, 0.5, 2.0)


def _point(**kwargs):
    return Point((1.0, 2.0, 3.0), id="p", description="d", props={"n": 4}, **kwargs)


class TestLayers(unittest.TestCase):
    def setUp(self):
        self.space = CoordinateSpace(SCALE)

    def test_image_and_segmentation(self):
        self.assertEqual(ImageLayer("zarr://x").to_json(), {
            "type": "image", "name": "image", "source": "zarr://x",
        })
        layer = SegmentationLayer("precomputed://s", segments=[1, 2]).to_json()
        self.assertEqual(layer["segments"], ["1", "2"])
        self.assertEqual(layer["skeletonRendering"]["mode2d"], "lines_and_points")
        atlas = SegmentationLayer("precomputed://ccf", name="ccf", skeleton_shader=None,
                                  extra={"objectAlpha": 0.2}).to_json()
        self.assertNotIn("skeletonRendering", atlas)
        self.assertEqual(atlas["objectAlpha"], 0.2)

    def test_annotation_layer_link_and_affine(self):
        linked = AnnotationLayer("precomputed://a", "m", link=("neuron", "neurons")).to_json(self.space)
        self.assertEqual(linked["source"], "precomputed://a")
        self.assertEqual(linked["linkedSegmentationLayer"], {"neuron": "neurons"})
        self.assertEqual(linked["filterBySegmentation"], ["neuron"])
        self.assertNotIn("linkedSegmentationLayer", AnnotationLayer("x", "m").to_json(self.space))

        moved = CoordinateSpace(SCALE, affine=[[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0]])
        source = AnnotationLayer("precomputed://a", "m").to_json(moved)["source"]
        self.assertEqual(source["url"], "precomputed://a")
        self.assertEqual(source["transform"]["matrix"][0], [1.0, 0.0, 0.0, 5.0])

    def test_local_annotation_layer(self):
        layer = LocalAnnotationLayer(
            "review", [_point(segments={"neuron": 3})], properties=[PropertySpec("n", "uint32")],
            link=("neuron", "neurons"),
        ).to_json(self.space)
        self.assertEqual(layer["source"], {
            "url": "local://annotations",
            "transform": {"outputDimensions": self.space.dimensions_json()},
        })
        self.assertEqual(layer["annotations"], [{
            "type": "point", "id": "p", "point": [1.0, 2.0, 3.0], "description": "d",
            "props": [4], "segments": [["3"]],
        }])
        self.assertEqual(layer["annotationRelationships"], ["neuron"])
        unlinked = LocalAnnotationLayer("r", [_point()], [PropertySpec("n", "uint32")]) \
            .to_json(self.space)
        self.assertNotIn("segments", unlinked["annotations"][0])
        self.assertNotIn("linkedSegmentationLayer", unlinked)

    def test_shaders(self):
        self.assertIn("if (n_flags > 0.5)", highlight_shader("n_flags"))
        self.assertIn("emitRGB(vec3(1.0, 0.0, 0.0));", highlight_shader("n_flags"))
        shader = compartment_shader()
        self.assertEqual(shader.count("emitRGB"), 4)
        self.assertIn("abs(compartment - 2.0) < 0.5", shader)
        self.assertNotIn("if", compartment_shader(colors={}))


class TestViewerState(unittest.TestCase):
    def test_build(self):
        view = (
            ViewerState(SCALE)
            .add(ImageLayer("zarr://img"))
            .add(SegmentationLayer("precomputed://s", segments=[1]))
            .add({"type": "annotation", "name": "raw", "source": "local://annotations"})
            .set_position((1, 2, 3))
            .select("neurons")
        )
        state = view.to_json()
        self.assertEqual(list(state), ["dimensions", "layers", "layout", "position", "selectedLayer"])
        self.assertEqual(view.layer_names(), ["image", "neurons", "raw"])
        self.assertEqual(state["position"], [1.0, 2.0, 3.0])
        self.assertEqual(state["layout"], "4panel")

    def test_position_follows_affine(self):
        space = CoordinateSpace((1, 1, 1), affine=[[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
        self.assertEqual(ViewerState(space).set_position((1, 1, 1)).to_json()["position"],
                         [2.0, 1.0, 2.0])

    def test_url(self):
        url = ViewerState(SCALE).to_url("https://example.org/ng/")
        self.assertTrue(url.startswith("https://example.org/ng/#!%7B"))
        fragment = url.split("#!", 1)[1]
        self.assertEqual(json.loads(urllib.parse.unquote(fragment))["layout"], "4panel")

    @needs_neuroglancer
    def test_url_matches_neuroglancer(self):
        state = ViewerState(SCALE).add(SegmentationLayer("precomputed://s", segments=[1])) \
            .set_position((1, 2, 3)).to_json()
        parsed = neuroglancer.parse_url(state_to_url(state))
        self.assertEqual(list(parsed.layers["neurons"].segments), [1])
        np.testing.assert_allclose(parsed.position, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
