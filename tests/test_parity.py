"""``QCReviewExporter`` reproduces the original standard_morph.neuroglancer output.

The fixtures in ``fixtures/golden`` were written by the original implementation
(see ``fixtures/golden/make_golden.py``). Changing the layers or links on
purpose means regenerating them, or replacing this test.
"""
import filecmp
import json
import unittest
from pathlib import Path

from neuron_ng.recipes.qc_review import QCReviewExporter
from tests.helpers import TmpCase, needs_neuroglancer

GOLDEN = Path(__file__).parent / "fixtures" / "golden"


class TestParity(TmpCase):
    def setUp(self):
        super().setUp()
        self.expected = json.loads((GOLDEN / "expected" / "states.json").read_text())
        self.exporter = QCReviewExporter(scale_um=self.expected["scale_um"])
        for name in self.expected["swcs"]:
            report = GOLDEN / "reports" / f"{Path(name).stem}_qc.json"
            self.exporter.add(GOLDEN / "swcs" / name, report)

    def test_states_and_urls(self):
        e, exporter = self.expected, self.exporter
        self.assertEqual([r.label for r in exporter.records], [n["label"] for n in e["neurons"]])
        self.assertEqual(exporter.batch_state(e["source"], image_source=e["image_source"]),
                         e["batch_state"])
        self.assertEqual(
            exporter.batch_url(e["source"], image_source=e["image_source"], base_url=e["base_url"]),
            e["batch_url"],
        )
        for neuron in e["neurons"]:
            with self.subTest(label=neuron["label"]):
                sid = neuron["segment_id"]
                self.assertEqual(exporter.neuron_state(sid, e["source"]), neuron["state"])
                self.assertEqual(exporter.neuron_url(sid, e["source"], base_url=e["base_url"]),
                                 neuron["url"])

    @needs_neuroglancer
    def test_layers_byte_identical(self):
        out = self.tmp / "layers"
        self.assertEqual(self.exporter.write(out), self.expected["written"])
        expected_dir = GOLDEN / "expected" / "layers"
        expected = sorted(p.relative_to(expected_dir) for p in expected_dir.rglob("*") if p.is_file())
        actual = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
        self.assertEqual(actual, expected)
        different = [str(p) for p in expected
                     if not filecmp.cmp(expected_dir / p, out / p, shallow=False)]
        self.assertEqual(different, [])


if __name__ == "__main__":
    unittest.main()
