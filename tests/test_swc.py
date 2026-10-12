import unittest

import numpy as np

from neuron_ng.io.swc import SWCError, read_swc, read_swc_table
from tests.helpers import LONG_EDGE_SWC, TmpCase


class TestReadSWC(TmpCase):
    def test_reads_nodes_edges_and_soma(self):
        skeleton = read_swc(self.swc("a.swc", LONG_EDGE_SWC))
        np.testing.assert_array_equal(skeleton.node_id, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(skeleton.xyz[:, 0], [0, 10, 20, 100, 110])
        self.assertEqual(skeleton.edges.tolist(), [[1, 0], [2, 1], [3, 2], [4, 3]])
        self.assertEqual(skeleton.soma_index, 0)
        np.testing.assert_array_equal(skeleton.compartment, [1, 2, 2, 2, 2])
        self.assertEqual(skeleton.home_position(), (0.0, 0.0, 0.0))

    def test_noncontiguous_ids_comments_and_stray_soma(self):
        text = (
            "# header\n"
            "10 3 0 0 0 1 -1\n"      # a root, but not a soma
            "30 1 5 5 5 1 10  # a soma-typed node mid-tree is not the soma\n"
            "20 1 7 7 7 2 -1\n"      # the soma: type 1 and a root
        )
        skeleton = read_swc(self.swc("a.swc", text))
        np.testing.assert_array_equal(skeleton.node_id, [10, 30, 20])
        self.assertEqual(skeleton.edges.tolist(), [[1, 0]])
        self.assertEqual(skeleton.soma_index, 2)
        self.assertEqual(skeleton.home_position(), (7.0, 7.0, 7.0))

    def test_missing_parent_gets_no_edge(self):
        skeleton = read_swc(self.swc("a.swc", LONG_EDGE_SWC + "6 3 1 2 3 1 99\n"))
        self.assertEqual(len(skeleton), 6)
        self.assertEqual(len(skeleton.edges), 4)

    def test_no_soma_centers_on_first_node(self):
        skeleton = read_swc(self.swc("a.swc", "1 2 4 5 6 1 -1\n2 2 7 8 9 1 1\n"))
        self.assertIsNone(skeleton.soma_index)
        self.assertEqual(skeleton.home_position(), (4.0, 5.0, 6.0))

    def test_offset_header(self):
        text = "# OFFSET 50 50 50\n# COLOR 1,1,1\n1 1 0 0 0 1 -1\n2 2 10 10 10 1 1\n"
        skeleton = read_swc(self.swc("a.swc", text))
        np.testing.assert_array_equal(skeleton.xyz, [[50, 50, 50], [60, 60, 60]])

    def test_bad_offset_header(self):
        with self.assertRaises(SWCError):
            read_swc(self.swc("a.swc", "# OFFSET 1 2\n1 1 0 0 0 1 -1\n"))

    def test_non_numeric_values(self):
        path = self.swc("a.swc", "1 1 0 0 0 1 -1\n2 2 abc 0 0 1 1\n")
        self.assertTrue(np.isnan(read_swc_table(path)["x"][1]))
        with self.assertRaises(SWCError):
            read_swc(path)

    def test_empty_file(self):
        skeleton = read_swc(self.swc("a.swc", "# nothing\n"))
        self.assertEqual(len(skeleton), 0)
        self.assertIsNone(skeleton.home_position())


if __name__ == "__main__":
    unittest.main()
