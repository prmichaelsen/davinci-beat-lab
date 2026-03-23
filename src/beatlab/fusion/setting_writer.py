"""Fusion .setting file serialization."""

from __future__ import annotations

from beatlab.fusion.nodes import FusionNode


class FusionComp:
    """A Fusion composition that can be serialized to a .setting file."""

    def __init__(self) -> None:
        self.nodes: list[FusionNode] = []
        self.active_tool: str | None = None

    def add_node(self, node: FusionNode) -> None:
        self.nodes.append(node)
        if self.active_tool is None:
            self.active_tool = node.name

    def serialize(self) -> str:
        """Produce the complete .setting file content."""
        lines = ["{", "    Tools = ordered() {"]

        for node in self.nodes:
            lines.append(node.to_lua())

            # Add BezierSpline tools for animated parameters
            for _spline_name, spline_lua in node.get_spline_tools():
                lines.append(spline_lua)

        lines.append("    },")

        if self.active_tool:
            lines.append(f'    ActiveTool = "{self.active_tool}",')

        lines.append("}")
        return "\n".join(lines) + "\n"

    def save(self, path: str) -> None:
        """Write the .setting file to disk."""
        with open(path, "w") as f:
            f.write(self.serialize())
