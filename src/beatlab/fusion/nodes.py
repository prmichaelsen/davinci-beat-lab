"""Fusion node type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from beatlab.fusion.keyframes import KeyframeTrack


@dataclass
class FusionNode:
    """Base class for a Fusion tool/node."""

    name: str
    tool_type: str
    inputs: dict[str, str | float | dict] = field(default_factory=dict)
    animated: dict[str, KeyframeTrack] = field(default_factory=dict)
    pos_x: float = 0
    pos_y: float = 0

    def get_spline_tools(self) -> list[tuple[str, str]]:
        """Return (spline_name, lua_block) pairs for animated params."""
        tools = []
        for param_name, track in self.animated.items():
            spline_name = f"{self.name}{param_name}"
            entries = track.to_lua_entries()
            kf_block = "\n".join(f"                {e}" for e in entries)
            lua = (
                f"        {spline_name} = BezierSpline {{\n"
                f"            SplineColor = {{ Red = 204, Green = 0, Blue = 0, }},\n"
                f"            KeyFrames = {{\n"
                f"{kf_block}\n"
                f"            }},\n"
                f"        }},"
            )
            tools.append((spline_name, lua))
        return tools

    def to_lua(self) -> str:
        """Serialize the node to Fusion .setting format."""
        lines = [f"        {self.name} = {self.tool_type} {{"]
        lines.append("            Inputs = {")

        for param_name, value in self.inputs.items():
            if isinstance(value, dict):
                # Connection: {"SourceOp": "name", "Source": "Output"}
                lines.append(
                    f'                {param_name} = Input {{ SourceOp = "{value["SourceOp"]}", '
                    f'Source = "{value["Source"]}", }},'
                )
            elif isinstance(value, (int, float)):
                lines.append(
                    f"                {param_name} = Input {{ Value = {value}, }},"
                )
            elif isinstance(value, str):
                lines.append(
                    f'                {param_name} = Input {{ Value = "{value}", }},'
                )

        # Animated params reference their BezierSpline
        for param_name in self.animated:
            spline_name = f"{self.name}{param_name}"
            lines.append(
                f'                {param_name} = Input {{ SourceOp = "{spline_name}", '
                f'Source = "Value", }},'
            )

        lines.append("            },")
        lines.append(
            f"            ViewInfo = OperatorInfo {{ Pos = {{ {self.pos_x}, {self.pos_y} }}, }},"
        )
        lines.append("        },")
        return "\n".join(lines)


def make_transform(
    name: str = "Transform1",
    source_op: str | None = None,
    pos_x: float = 110,
) -> FusionNode:
    """Create a Transform node."""
    inputs: dict = {}
    if source_op:
        inputs["Input"] = {"SourceOp": source_op, "Source": "Output"}
    return FusionNode(name=name, tool_type="Transform", inputs=inputs, pos_x=pos_x)


def make_brightness_contrast(
    name: str = "BrightnessContrast1",
    source_op: str | None = None,
    pos_x: float = 220,
) -> FusionNode:
    """Create a BrightnessContrast node."""
    inputs: dict = {}
    if source_op:
        inputs["Input"] = {"SourceOp": source_op, "Source": "Output"}
    return FusionNode(
        name=name, tool_type="BrightnessContrast", inputs=inputs, pos_x=pos_x,
    )


def make_glow(
    name: str = "Glow1",
    source_op: str | None = None,
    pos_x: float = 330,
) -> FusionNode:
    """Create a Glow node."""
    inputs: dict = {}
    if source_op:
        inputs["Input"] = {"SourceOp": source_op, "Source": "Output"}
    return FusionNode(name=name, tool_type="Glow", inputs=inputs, pos_x=pos_x)
