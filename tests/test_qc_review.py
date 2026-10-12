"""Tests for the QC review recipe (ported from standard_morph.neuroglancer)."""
import json
import unittest

import numpy as np

from neuron_ng.adapters.standard_morph_qc import FlaggedNode
from neuron_ng.core.coords import CoordinateSpace
from neuron_ng.recipes.qc_review import (
    ANNOTATION_DIR, SKELETON_DIR, QCReviewExporter, review_layer,
)
from neuron_ng.state import ViewerState
from tests.helpers import (
    CLEAN_SWC, LONG_EDGE_SWC, TmpCase, decode_skeleton, long_edge_report, needs_neuroglancer,
    neuroglancer, report, result,
)

SOURCE = "s3://bucket/prefix"
SCALE = (0.5, 0.5, 2.0)


class _ExporterCase(TmpCase):
    def setUp(self):
        super().setUp()
        self.long_edge = self.swc("long_edge.swc", LONG_EDGE_SWC)
        self.clean = self.swc("clean.swc", CLEAN_SWC)
        self.exporter = QCReviewExporter(SCALE)


class TestRecords(_ExporterCase):
    def test_add_collects_flagged_nodes(self):
        segment_id = self.exporter.add(self.long_edge, long_edge_report())
        record = self.exporter.record(segment_id)
        self.assertEqual(segment_id, 1)
        self.assertEqual(record.label, "long_edge")
        self.assertEqual(self.exporter.metrics, ["edge_length"])
        self.assertEqual([n.node_id for n in record.flags["edge_length"]], [4])
        self.assertEqual(record.position, (100.0, 0.0, 0.0))

    def test_label_from_input_ref_then_argument(self):
        self.exporter.add(self.long_edge, long_edge_report(input_ref="/data/N001.swc"))
        self.exporter.add(self.long_edge, long_edge_report(), label="mine")
        self.assertEqual([r.label for r in self.exporter.records], ["N001", "mine"])

    def test_clean_neuron_is_centered_on_soma(self):
        self.exporter.add(self.clean, report())
        record = self.exporter.record(1)
        self.assertEqual(record.flags, {})
        self.assertEqual(record.position, (5.0, 6.0, 7.0))

    def test_unbuildable_file_has_no_skeleton(self):
        bad = self.swc("bad.swc", "1 1 0 0 0 1 -1\n2 2 abc 0 0 1 1\n")
        self.exporter.add(bad, report(
            integrity=[result("castable_columns", "fail", [2])], morphology_evaluated=False,
        ))
        record = self.exporter.record(1)
        self.assertIsNone(record.skeleton)
        self.assertEqual(record.flags, {})  # node 2 has no coordinates to place it
        self.assertIsNone(record.position)

    def test_nodes_without_coordinates_placed_from_skeleton(self):
        dangling = self.swc("dangling.swc", LONG_EDGE_SWC + "6 3 1 2 3 1 99\n")
        self.exporter.add(dangling, report(integrity=[result("valid_parent_references", "fail", [6])]))
        nodes = self.exporter.record(1).flags["valid_parent_references"]
        self.assertEqual([(n.node_id, n.xyz, n.status) for n in nodes], [(6, (1.0, 2.0, 3.0), "fail")])

    def test_needs_exactly_one_of_scale_and_space(self):
        for kwargs in ({}, {"scale_um": SCALE, "space": CoordinateSpace(SCALE)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QCReviewExporter(**kwargs)
        self.assertEqual(QCReviewExporter(space=CoordinateSpace(SCALE)).scale_um, SCALE)


class TestWrite(_ExporterCase):
    @needs_neuroglancer
    def test_skeleton_flags_and_segment_properties(self):
        self.exporter.add(self.long_edge, long_edge_report())
        self.exporter.add(self.clean, report())
        written = self.exporter.write(self.tmp / "ng")
        self.assertEqual(written, {"skeletons": [1, 2], "annotations": ["edge_length"]})

        skeleton_dir = self.tmp / "ng" / SKELETON_DIR
        info = json.loads((skeleton_dir / "info").read_text())
        names = [a["id"] for a in info["vertex_attributes"]]
        self.assertEqual(names, ["radius", "n_flags", "flag_edge_length"])
        _, _, attributes = decode_skeleton((skeleton_dir / "1").read_bytes(), names)
        np.testing.assert_array_equal(attributes["flag_edge_length"], [0, 0, 0, 1, 0])
        np.testing.assert_array_equal(attributes["n_flags"], [0, 0, 0, 1, 0])

        props = json.loads((skeleton_dir / "segment_properties" / "info").read_text())["inline"]
        label, tags, counts = props["properties"]
        self.assertEqual(label["values"], ["long_edge", "clean"])
        self.assertEqual((tags["tags"], tags["values"]), (["edge_length"], [[0], []]))
        self.assertEqual((counts["id"], counts["values"]), ("n_flagged_edge_length", [1, 0]))

        annotations = self.tmp / "ng" / ANNOTATION_DIR / "edge_length"
        info = json.loads((annotations / "info").read_text())
        self.assertEqual([p["id"] for p in info["properties"]], ["node_id", "status"])
        self.assertTrue((annotations / "rel_neuron" / "1").exists())

    def test_no_flags_writes_skeleton_without_annotations(self):
        self.exporter.add(self.clean, report())
        self.assertEqual(self.exporter.write(self.tmp / "ng"), {"skeletons": [1], "annotations": []})
        self.assertFalse((self.tmp / "ng" / ANNOTATION_DIR).exists())


class TestState(_ExporterCase):
    def setUp(self):
        super().setUp()
        self.exporter.add(self.long_edge, long_edge_report())
        self.exporter.add(self.clean, report())

    def test_batch_state(self):
        state = self.exporter.batch_state(SOURCE)
        self.assertEqual(state["dimensions"]["x"], [5e-07, "m"])
        neurons, annotations = state["layers"]
        self.assertEqual(neurons["source"], "precomputed://s3://bucket/prefix/skeletons")
        self.assertEqual(neurons["segments"], ["1", "2"])
        self.assertIn("n_flags", neurons["skeletonRendering"]["shader"])
        self.assertEqual(annotations["source"], "precomputed://s3://bucket/prefix/annotations/edge_length")
        self.assertEqual(annotations["linkedSegmentationLayer"], {"neuron": "neurons"})
        self.assertEqual(state["position"], [100.0, 0.0, 0.0])
        self.assertEqual(state["selectedLayer"], {"layer": "neurons", "visible": True})

    def test_image_layer_only_when_given(self):
        self.assertNotIn("image", [l["name"] for l in self.exporter.batch_state(SOURCE)["layers"]])
        state = self.exporter.batch_state(SOURCE, image_source="zarr://s3://bucket/image.zarr")
        self.assertEqual(state["layers"][0], {
            "type": "image", "name": "image", "source": "zarr://s3://bucket/image.zarr",
        })

    def test_neuron_state(self):
        flagged = self.exporter.neuron_state(1, SOURCE)
        self.assertEqual(flagged["layers"][0]["segments"], ["1"])
        self.assertEqual([l["name"] for l in flagged["layers"]], ["neurons", "qc_errors"])
        self.assertEqual(flagged["position"], [100.0, 0.0, 0.0])
        self.assertEqual(flagged["selectedLayer"], {"layer": "qc_errors", "visible": True})

        clean = self.exporter.neuron_state(2, SOURCE)
        self.assertEqual([l["name"] for l in clean["layers"]], ["neurons"])
        self.assertEqual(clean["position"], [5.0, 6.0, 7.0])

    def test_views_are_editable_builders(self):
        view = self.exporter.batch_view(SOURCE)
        self.assertIsInstance(view, ViewerState)
        view.add({"type": "annotation", "name": "notes", "source": "local://annotations"})
        self.assertEqual(view.layer_names()[-1], "notes")

    def test_review_layer_lists_flagged_nodes(self):
        review = self.exporter.neuron_state(1, SOURCE)["layers"][-1]
        self.assertEqual(review["source"]["url"], "local://annotations")
        self.assertEqual(review["tab"], "annotations")
        self.assertEqual(review["linkedSegmentationLayer"], {"neuron": "neurons"})
        self.assertEqual(review["annotations"], [{
            "type": "point", "id": "edge_length-4", "point": [100.0, 0.0, 0.0],
            "description": "edge_length · fail · node 4", "props": [4, 0, 0],
            "segments": [["1"]],
        }])

    def test_review_layer_without_skeleton_is_unlinked(self):
        flags = {"edge_length": [FlaggedNode(4, (1.0, 2.0, 3.0), "review")]}
        review = review_layer(flags).to_json(CoordinateSpace(SCALE))
        self.assertNotIn("linkedSegmentationLayer", review)
        self.assertNotIn("annotationRelationships", review)
        self.assertEqual(review["annotations"][0]["props"], [4, 1, 0])
        self.assertNotIn("segments", review["annotations"][0])

    def test_review_layer_dedupes_nodes_per_metric(self):
        node = FlaggedNode(4, (1.0, 2.0, 3.0), "fail")
        review = review_layer({"a": [node, node], "b": [node]}).to_json(CoordinateSpace(SCALE))
        self.assertEqual([a["id"] for a in review["annotations"]], ["a-4", "b-4"])

    def test_url_prefix(self):
        url = self.exporter.batch_url(SOURCE, base_url="https://example.org/ng/")
        self.assertTrue(url.startswith("https://example.org/ng/#!%7B"))

    @needs_neuroglancer
    def test_url_round_trips_through_neuroglancer(self):
        url = self.exporter.batch_url(SOURCE, image_source="zarr://s3://bucket/image.zarr")
        parsed = neuroglancer.parse_url(url)
        self.assertEqual([l.name for l in parsed.layers], ["image", "neurons", "edge_length"])
        self.assertEqual(list(parsed.layers["neurons"].segments), [1, 2])
        np.testing.assert_allclose(parsed.position, [100.0, 0.0, 0.0])

    @needs_neuroglancer
    def test_neuron_url_review_layer_round_trips(self):
        parsed = neuroglancer.parse_url(self.exporter.neuron_url(1, SOURCE))
        review = parsed.layers["qc_errors"]
        self.assertEqual(review.source[0].url, "local://annotations")
        (point,) = review.annotations
        self.assertEqual(point.description, "edge_length · fail · node 4")
        np.testing.assert_allclose(point.point, [100.0, 0.0, 0.0])
        self.assertEqual([int(s) for s in point.segments[0]], [1])
        self.assertEqual([p.id for p in review.annotation_properties], ["node_id", "status", "metric"])


if __name__ == "__main__":
    unittest.main()
