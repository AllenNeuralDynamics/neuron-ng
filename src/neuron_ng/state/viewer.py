"""Build Neuroglancer viewer states and links.

States are plain JSON dicts, so building a link does not need the
``neuroglancer`` package::

    state = (
        ViewerState(CoordinateSpace((0.748, 0.748, 1.0)))
        .add(ImageLayer("zarr://s3://bucket/fused.zarr"))
        .add(SegmentationLayer("precomputed://s3://bucket/run/skeletons", segments=[1, 2]))
        .set_position((100, 200, 300))
    )
    url = state.to_url()
"""
import json
import urllib.parse

from neuron_ng.core.coords import CoordinateSpace

DEFAULT_BASE_URL = "https://neuroglancer-demo.appspot.com"

# Characters left unescaped in the URL fragment, as neuroglancer.to_url does.
_URL_SAFE = "~@#$&()*!+=:;,.?/'"


class ViewerState:
    """A viewer state in ``space``: its dimensions, layers (bottom first),
    position, layout and selected layer.

    Positions are given in reconstruction units and mapped through the space's
    affine.
    """

    def __init__(self, space, layout="4panel"):
        self.space = CoordinateSpace.coerce(space)
        self.layout = layout
        self.layers = []
        self.position = None
        self.selected_layer = None
        self.extra = {}

    def add(self, layer):
        """Append a layer spec (anything with ``to_json(space)``) or a raw
        layer dict. Returns ``self``."""
        self.layers.append(layer)
        return self

    def set_position(self, xyz):
        self.position = None if xyz is None else self.space.apply(xyz)
        return self

    def select(self, name, visible=True):
        """Open layer ``name`` in the side panel on load."""
        self.selected_layer = {"layer": name, "visible": visible}
        return self

    def layer_names(self):
        return [layer_json["name"] for layer_json in self._layers_json()]

    def _layers_json(self):
        return [
            layer if isinstance(layer, dict) else layer.to_json(self.space)
            for layer in self.layers
        ]

    def to_json(self):
        state = {
            "dimensions": self.space.dimensions_json(),
            "layers": self._layers_json(),
            "layout": self.layout,
        }
        if self.position is not None:
            state["position"] = [float(v) for v in self.position]
        if self.selected_layer is not None:
            state["selectedLayer"] = dict(self.selected_layer)
        state.update(self.extra)
        return state

    def to_url(self, base_url=DEFAULT_BASE_URL):
        return state_to_url(self.to_json(), base_url)


def state_to_url(state, base_url=DEFAULT_BASE_URL):
    """Encode a state dict as a Neuroglancer link (the same encoding as
    ``neuroglancer.to_url``)."""
    fragment = urllib.parse.quote(json.dumps(state, separators=(",", ":")), safe=_URL_SAFE)
    return f"{base_url.rstrip('/')}/#!{fragment}"
