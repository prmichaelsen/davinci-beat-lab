# Fusion .setting File Format Reference

**Created**: 2026-03-23
**Status**: Active

## Overview

Fusion .setting files use **Lua table serialization**. The file is a single Lua table literal containing an `ordered()` Tools table with node definitions and BezierSpline modifiers for keyframes.

## Top-Level Structure

```lua
{
    Tools = ordered() {
        ToolName = ToolType {
            Inputs = { ... },
            ViewInfo = OperatorInfo { Pos = { x, y }, },
        },
    },
    ActiveTool = "ToolName",
}
```

## Node Types

| Node | Tool Type ID | Key Inputs |
|------|-------------|------------|
| Transform | `Transform` | Size, Center, Angle, Input |
| BrightnessContrast | `BrightnessContrast` | Gain, Brightness, Contrast, Input |
| Glow | `Glow` | Glow, GlowSize, Input |
| MediaIn | `MediaIn` | (none) |
| MediaOut | `MediaOut` | Input |

## Input Types

**Static value:**
```lua
Gain = Input { Value = 1.2, },
```

**Connection to another node:**
```lua
Input = Input { SourceOp = "Transform1", Source = "Output", },
```

**Connection to spline (keyframed):**
```lua
Size = Input { SourceOp = "Transform1Size", Source = "Value", },
```

## Keyframes (BezierSpline)

Keyframed parameters use a separate `BezierSpline` tool referenced by `SourceOp`:

```lua
Transform1Size = BezierSpline {
    SplineColor = { Red = 204, Green = 0, Blue = 0, },
    KeyFrames = {
        [0] = { 1.0, RH = { 4.0, 1.0 }, Flags = { Linear = true }, },
        [12] = { 1.1, LH = { 8.0, 1.1 }, RH = { 16.0, 1.1 }, },
        [24] = { 1.0, LH = { 20.0, 1.0 }, Flags = { Linear = true }, },
    },
},
```

**Keyframe format:** `[frame] = { value, LH = { lh_x, lh_y }, RH = { rh_x, rh_y }, Flags = { ... }, }`
- `LH`/`RH` handles are **absolute coordinates** (frame-space X, value-space Y)
- First keyframe: only RH meaningful. Last: only LH.
- Without LH/RH or with `Linear = true`: linear interpolation

## Interpolation Flags

| Flag | Meaning |
|------|---------|
| `Linear = true` | Linear interpolation |
| `StepIn = true` | Hold previous value until this frame |
| `StepOut = true` | Jump to value immediately |
| (no flags + handles) | Bezier/smooth interpolation |

## Easing via Handles

- **Ease-out**: RH.y matches keyframe value (flat tangent leaving)
- **Ease-in**: LH.y matches keyframe value (flat tangent arriving)
- **Smooth**: Both handles have y matching value (S-curve)
- Handle X position controls curve tension (closer = sharper, farther = smoother)
