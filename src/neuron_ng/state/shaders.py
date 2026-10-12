"""Shader snippets and palettes for skeleton and annotation layers."""

#: Colors each skeleton segment by its own (random) segment color.
DEFAULT_SKELETON_SHADER = """void main() {
  emitDefault();
}
"""

#: Colors cycled over annotation layers.
ANNOTATION_COLORS = (
    "#ffff00", "#00ffff", "#ff00ff", "#ff8000", "#00ff80",
    "#8080ff", "#ff0080", "#80ff00", "#0080ff", "#ffffff",
)

#: SWC compartment colors: soma, axon, basal and apical dendrite.
COMPARTMENT_COLORS = {
    1: (1.0, 1.0, 1.0),
    2: (0.25, 0.5, 1.0),
    3: (1.0, 0.25, 0.25),
    4: (1.0, 0.25, 1.0),
}


def _vec3(rgb):
    return "vec3({}, {}, {})".format(*(float(v) for v in rgb))


def highlight_shader(attribute, rgb=(1.0, 0.0, 0.0), threshold=0.5):
    """Draws vertices whose ``attribute`` exceeds ``threshold`` in ``rgb``, and
    everything else in the segment's own color."""
    return (
        "void main() {\n"
        f"  if ({attribute} > {float(threshold)}) {{\n"
        f"    emitRGB({_vec3(rgb)});\n"
        "  } else {\n"
        "    emitDefault();\n"
        "  }\n"
        "}\n"
    )


def compartment_shader(attribute="compartment", colors=None):
    """Colors vertices by SWC compartment code (a float vertex attribute).
    Codes without a color use the segment's own color."""
    colors = COMPARTMENT_COLORS if colors is None else colors
    lines = ["void main() {"]
    for i, (code, rgb) in enumerate(colors.items()):
        keyword = "if" if i == 0 else "} else if"
        lines.append(f"  {keyword} (abs({attribute} - {float(code)}) < 0.5) {{")
        lines.append(f"    emitRGB({_vec3(rgb)});")
    if colors:
        lines += ["  } else {", "    emitDefault();", "  }"]
    else:
        lines.append("  emitDefault();")
    lines.append("}")
    return "\n".join(lines) + "\n"


def point_shader(size=10.0):
    """Annotation shader: the layer color at a fixed marker size."""
    return (
        "void main() {\n"
        "  setColor(defaultColor());\n"
        f"  setPointMarkerSize({float(size)});\n"
        "}\n"
    )


DEFAULT_ANNOTATION_SHADER = point_shader()
