# neuron-ng

Neuroglancer layers and views for neuron reconstructions.

`neuron_ng` writes reconstructions (SWCs) as [Neuroglancer](https://github.com/google/neuroglancer) precomputed sources and builds viewer links to them. The core is generic: skeletons, typed point annotations, segment properties, and a viewer-state builder. Ready-made **recipes** build on it for specific uses:

- `recipes.qc_review`: shows the nodes a [standard_morph](https://github.com/AllenNeuralDynamics/standard_morph) QC run flagged, with a batch link and a per-neuron review link. It reads the QC report JSON, so standard_morph does not have to be installed.
- `recipes.gallery`: many neurons in one source, with labels, tags and numbers to search by, over an optional image and atlas.

## Install

```
pip install "neuron-ng[annotations,s3] @ git+https://github.com/AllenNeuralDynamics/neuron-ng"
```

| Extra | Needed for |
|-------|------------|
| (none) | Reading SWCs, writing skeletons and segment properties, building states and links. Needs only `numpy`. |
| `annotations` | Writing precomputed point annotations (the `neuroglancer` package, Python >= 3.10). |
| `s3` | `neuron_ng.storage.upload_dir` (`boto3`). |

## QC review

```python
from neuron_ng.recipes.qc_review import QCReviewExporter
from neuron_ng.storage import upload_dir

exporter = QCReviewExporter(scale_um=(0.748, 0.748, 1.0))
for swc, report in runs:           # report: RunReport, its to_dict(), or a *_qc.json path
    segment_id = exporter.add(swc, report)
exporter.write("out/ng")
upload_dir("out/ng", "s3://bucket/qc/run1")

batch_url = exporter.batch_url("s3://bucket/qc/run1", image_source="zarr://s3://bucket/fused.zarr")
neuron_url = exporter.neuron_url(1, "s3://bucket/qc/run1")
```

**Layers:**
- `skeletons/` has one segment per neuron. Each vertex has `radius`, `n_flags` (how many metrics flagged it) and `flag_<metric>`. The default shader draws flagged nodes in red. Each segment is tagged with its failing metrics and has `n_flagged_<metric>` counts.
- `annotations/<metric>/` has a point at each node that metric flagged, with the SWC `node_id` and the result `status`, related to the neuron's segment.

**Links:**
- The batch link shows every neuron and one annotation layer per metric, filtered to the selected neurons.
- A neuron's link is centered on its first flagged node (or its soma) and lists its errors in an editable `qc_errors` layer. Click one to jump to it. Edit its description or delete it to track what you've fixed. Edits live only in the link, so save the updated URL.

**Report contract:** `adapters.standard_morph_qc` reads `RunReport.to_dict()` (schema `1.x`, and warns on other versions). It uses `integrity_results` and `results` (`name`, `status`, `flagged_node_ids`, `flagged_node_coordinates`), `morphology_evaluated` and `input_ref`, and ignores other keys. When `morphology_evaluated` is false the neuron has no skeleton, but flagged nodes with coordinates are still shown. Node ids reported without coordinates are placed from the SWC.

## Gallery

```python
from neuron_ng.recipes.gallery import GalleryExporter
from neuron_ng.state import SegmentationLayer

gallery = GalleryExporter(scale_um=(1, 1, 1))              # e.g. SWCs in CCF micrometers
for path in swc_paths:
    gallery.add(path, tags=["VISp"], numbers={"soma_depth_um": 412.0})
gallery.write("out/gallery")
url = gallery.url(
    "s3://bucket/gallery",
    extra_layers=[SegmentationLayer("precomputed://s3://bucket/ccf/annotation_10",
                                    name="ccf", skeleton_shader=None)],
)
```

## Coordinates

Layers are written in the SWC's own units and never converted. `scale_um` is the size of one unit in micrometers: `(1, 1, 1)` for SWCs in micrometers (including CCF), and the voxel size for SWCs in voxel indices. A wrong scale misaligns the layers with the image. For SWCs that are offset, flipped or rotated relative to their image, pass `space=CoordinateSpace(scale_um, affine=...)`, where the 3x4 affine maps SWC coordinates to viewer coordinates. It reaches skeletons through their `info` transform, annotation layers through source transforms, and positions directly.

## Building other views

Everything a recipe does is available on its own:

```python
from neuron_ng import CoordinateSpace, Neuron, ViewerState, read_swc
from neuron_ng.precomputed import write_skeleton_source
from neuron_ng.state import ImageLayer, SegmentationLayer

space = CoordinateSpace((1, 1, 1))
neurons = [Neuron(1, "manual", read_swc("manual.swc")), Neuron(2, "auto", read_swc("auto.swc"))]
write_skeleton_source(neurons, "out/compare/skeletons", space)
url = (
    ViewerState(space)
    .add(ImageLayer("zarr://s3://bucket/fused.zarr"))
    .add(SegmentationLayer("precomputed://s3://bucket/compare/skeletons", segments=[1, 2]))
    .set_position(neurons[0].skeleton.home_position())
    .to_url()
)
```

## Design

The package follows a ports-and-adapters layout. Dependencies point inward, so the core never knows about a particular tool or use case:

```
recipes/      qc_review, gallery           compose the layers below into products and links
adapters/     standard_morph_qc            the only code that knows another tool's format
state/        layers, viewer, shaders      viewer states as plain JSON (no neuroglancer needed)
precomputed/  skeletons, annotations, ...  writers for precomputed sources
io/           swc                          readers that build the model
core/         coords, model                CoordinateSpace, Skeleton, Neuron, Point, PropertySpec
```

- **Products vs. views.** `write()` produces data (precomputed sources) once. Views are cheap states that reference that data by URL, so one upload can back any number of links.
- **Adapters isolate formats.** Supporting another QC tool or proofreading log means adding a module under `adapters/` that produces core objects. It doesn't mean changing the writers.
- **Recipes are thin.** A new view (e.g. comparing reconstructions, or a CCF projection view) is a new module under `recipes/`. Recipes return `ViewerState` builders (`batch_view`, `neuron_view`, `view`), so callers can add layers before encoding.
- **Optional dependencies are gated.** Only annotation writing imports `neuroglancer`, and only `storage` imports `boto3`.

## Tests

```
pip install -e ".[test]"
python -m pytest
```

`tests/test_parity.py` checks that `QCReviewExporter` writes the same bytes and links as the original `standard_morph.neuroglancer` implementation it replaces, using fixtures from `tests/fixtures/golden/make_golden.py`.
