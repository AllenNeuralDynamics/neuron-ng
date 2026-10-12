import json
import unittest
from unittest import mock

from neuron_ng.adapters.standard_morph_qc import ReportError, load_report
from tests.helpers import TmpCase, report, result


class TestLoadReport(TmpCase):
    def test_reads_dict_path_and_to_dict(self):
        payload = report([result("edge_length", "fail", [4], [(1, 2, 3)])], input_ref="a.swc")
        path = self.tmp / "a_qc.json"
        path.write_text(json.dumps(payload))

        class RunReportLike:
            def to_dict(self):
                return payload

        for source in (payload, path, str(path), RunReportLike()):
            with self.subTest(source=type(source).__name__):
                qc = load_report(source)
                self.assertEqual(qc.input_ref, "a.swc")
                self.assertEqual([r.name for r in qc.results], ["edge_length"])

    def test_extra_keys_ignored(self):
        qc = load_report({"validation": {"passed": False}, **report()})
        self.assertTrue(qc.morphology_evaluated)

    def test_not_a_report(self):
        for bad in ({"error": {"type": "OSError"}}, [1, 2], 5):
            with self.subTest(bad=bad), self.assertRaises(ReportError):
                load_report(bad)
        path = self.tmp / "bad.json"
        path.write_text("{")
        with self.assertRaises(ReportError):
            load_report(path)

    def test_unknown_schema_version_warns(self):
        with self.assertLogs("neuron_ng.adapters.standard_morph_qc", level="WARNING"):
            load_report(report(schema_version="2.0"))
        with mock.patch("neuron_ng.adapters.standard_morph_qc.logger") as logger:
            load_report(report(schema_version="1.7"))
        logger.warning.assert_not_called()


class TestFlaggedNodes(unittest.TestCase):
    def test_integrity_first_and_statuses_filtered(self):
        qc = load_report(report(
            integrity=[result("valid_parent_references", "fail", [6], [(1, 2, 3)])],
            results=[
                result("edge_length", "review", [4], [(9, 9, 9)]),
                result("tortuosity", "pass", [5], [(0, 0, 0)]),
                result("skipped_one", "skipped", [5], [(0, 0, 0)]),
            ],
        ))
        flags = qc.flagged_nodes()
        self.assertEqual(list(flags), ["valid_parent_references", "edge_length"])
        self.assertEqual(flags["edge_length"][0].status, "review")
        self.assertEqual(list(qc.flagged_nodes(statuses=("fail",))), ["valid_parent_references"])

    def test_nodes_without_coordinates_placed_by_id(self):
        qc = load_report(report(integrity=[result("valid_parent_references", "fail", [6, 7])]))
        self.assertEqual(qc.flagged_nodes(), {})
        flags = qc.flagged_nodes(xyz_by_id={6: (1, 2, 3)})
        self.assertEqual([(n.node_id, n.xyz) for n in flags["valid_parent_references"]],
                         [(6, (1.0, 2.0, 3.0))])

    def test_non_finite_coordinates_dropped(self):
        qc = load_report(report([result("m", "fail", [1, 2], [(float("nan"), 0, 0), (1, 1, 1)])]))
        self.assertEqual([n.node_id for n in qc.flagged_nodes()["m"]], [2])


if __name__ == "__main__":
    unittest.main()
