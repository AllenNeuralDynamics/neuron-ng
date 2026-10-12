"""Write a precomputed segment-properties table for a set of neurons.

The table has one row per neuron: its ``label``, its ``tags`` (indices into a
shared vocabulary, so Neuroglancer can filter by tag), and one numeric property
per name in ``number_types``.
"""
import json
from pathlib import Path


def segment_properties_info(neurons, tag_vocabulary=(), number_types=None):
    """Return the inline ``neuroglancer_segment_properties`` info.

    Parameters
    ----------
    neurons : list[Neuron]
    tag_vocabulary : sequence of str
        Every tag that may appear, in display order. Neuron tags outside it are
        ignored. With an empty vocabulary there is still a (empty) tags property.
    number_types : dict[str, str], optional
        ``{name: data_type}`` (e.g. ``"uint32"``, ``"float32"``) for each
        numeric property, in order. A neuron without a value gets 0.
    """
    tags = list(tag_vocabulary)
    tag_index = {t: i for i, t in enumerate(tags)}
    properties = [
        {"id": "label", "type": "label", "values": [n.label for n in neurons]},
        {
            "id": "tags",
            "type": "tags",
            "tags": tags,
            "values": [sorted(tag_index[t] for t in n.tags if t in tag_index) for n in neurons],
        },
    ]
    for name, data_type in (number_types or {}).items():
        properties.append({
            "id": name,
            "type": "number",
            "data_type": data_type,
            "values": [n.numbers.get(name, 0) for n in neurons],
        })
    return {
        "@type": "neuroglancer_segment_properties",
        "inline": {"ids": [str(n.segment_id) for n in neurons], "properties": properties},
    }


def write_segment_properties(path, neurons, tag_vocabulary=(), number_types=None):
    """Write :func:`segment_properties_info` to ``path/info``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    info = segment_properties_info(neurons, tag_vocabulary, number_types)
    (path / "info").write_text(json.dumps(info))
