"""Lazy import of the optional ``neuroglancer`` dependency."""


def import_neuroglancer():
    try:
        import neuroglancer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Writing Neuroglancer annotations requires the 'neuroglancer' package "
            "(Python >= 3.10). Install it with: pip install 'neuron-ng[annotations]'"
        ) from exc
    return neuroglancer
