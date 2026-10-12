"""Regenerate the golden fixtures from the original implementation.

The expected outputs were produced by ``standard_morph.neuroglancer`` on
standard_morph's ``feat-neuroglancer`` branch (commit 2aef6f7 plus the
``qc_errors`` review layer), which this package replaces. They are kept so
``tests/test_parity.py`` can check that ``QCReviewExporter`` still writes the
same layers and links. Run it only with that version of standard_morph
installed (with its ``neuroglancer`` extra)::

    python tests/fixtures/golden/make_golden.py

It writes ``reports/``, ``expected/layers/`` and ``expected/states.json``
next to this file. The SWCs in ``swcs/`` are inputs and are not regenerated.
"""
import json
import shutil
from pathlib import Path

from standard_morph import MorphologyKind, Policy, QCContext, Space, run_qc
from standard_morph.neuroglancer import NeuroglancerExporter

HERE = Path(__file__).parent
SCALE = (0.748, 0.748, 1.0)
SOURCE = "s3://bucket/run"
IMAGE = "zarr://s3://bucket/image.zarr"
BASE_URL = "https://neuroglancer.example.org"
METRICS = [
    "single_root_node", "soma_first_node", "single_connected_component", "branch_max_degree",
    "edge_length", "soma_child_distance", "axon_origination", "local_tortuosity",
]
# Tight enough that the real reconstruction is flagged by several metrics.
THRESHOLDS = {
    "edge_length": {"max_length_um": 600.0},
    "soma_child_distance": {"max_soma_child_to_soma_um": 5.0},
    "axon_origination": {"max_axon_origin_to_soma_um": 5.0},
    "branch_max_degree": {"max_children": 2},
    "local_tortuosity": {"tortuosity_threshold": 3.0},
}


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main():
    context = QCContext(space=Space.IMAGE_SPACE, morphology_kind=MorphologyKind.MERGED)
    policy = Policy(version="golden", thresholds=THRESHOLDS)
    exporter = NeuroglancerExporter(SCALE)
    reports_dir = HERE / "reports"
    shutil.rmtree(reports_dir, ignore_errors=True)
    reports_dir.mkdir()
    swcs = sorted((HERE / "swcs").glob("*.swc"))
    for swc in swcs:
        report = run_qc(str(swc), context, metrics=METRICS, policy=policy)
        payload = report.to_dict()
        # Store a path relative to the fixtures, so the default label is stable.
        payload["input_ref"] = f"swcs/{swc.name}"
        (reports_dir / f"{swc.stem}_qc.json").write_text(
            json.dumps(payload, indent=1, default=_json_default) + "\n"
        )
        exporter.add(str(swc), report)

    expected = HERE / "expected"
    shutil.rmtree(expected, ignore_errors=True)
    written = exporter.write(expected / "layers")
    states = {
        "scale_um": list(SCALE),
        "source": SOURCE,
        "image_source": IMAGE,
        "base_url": BASE_URL,
        "swcs": [swc.name for swc in swcs],
        "written": written,
        "batch_state": exporter.batch_state(SOURCE, image_source=IMAGE),
        "batch_url": exporter.batch_url(SOURCE, image_source=IMAGE, base_url=BASE_URL),
        "neurons": [
            {
                "segment_id": r.segment_id,
                "label": r.label,
                "state": exporter.neuron_state(r.segment_id, SOURCE),
                "url": exporter.neuron_url(r.segment_id, SOURCE, base_url=BASE_URL),
            }
            for r in exporter.records
        ],
    }
    (expected / "states.json").write_text(json.dumps(states, indent=1) + "\n")
    print(f"Wrote {len(swcs)} reports and {written} to {HERE}")


if __name__ == "__main__":
    main()
