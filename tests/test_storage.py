import tempfile
import unittest
from pathlib import Path
from unittest import mock

from neuron_ng.storage import content_type, parse_s3_uri, upload_dir


class TestUpload(unittest.TestCase):
    def test_parse_s3_uri(self):
        self.assertEqual(parse_s3_uri("s3://bucket/a/b/"), ("bucket", "a/b"))
        self.assertEqual(parse_s3_uri("s3://bucket"), ("bucket", ""))
        for bad in ("bucket/a", "s3:///a"):
            with self.subTest(uri=bad), self.assertRaises(ValueError):
                parse_s3_uri(bad)

    def test_content_type(self):
        self.assertEqual(content_type("skeletons/info"), "application/json")
        self.assertEqual(content_type("skeletons/1"), "application/octet-stream")

    def test_upload_dir_keeps_relative_paths(self):
        client = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "skeletons").mkdir()
            (Path(tmp) / "skeletons" / "info").write_text("{}")
            (Path(tmp) / "skeletons" / "1").write_bytes(b"x")
            self.assertEqual(upload_dir(tmp, "s3://bucket/run/", client=client), 2)
        calls = sorted((c.args[1], c.args[2], c.kwargs["ExtraArgs"]["ContentType"])
                       for c in client.upload_file.call_args_list)
        self.assertEqual(calls, [
            ("bucket", "run/skeletons/1", "application/octet-stream"),
            ("bucket", "run/skeletons/info", "application/json"),
        ])


if __name__ == "__main__":
    unittest.main()
