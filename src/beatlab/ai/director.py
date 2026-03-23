"""AI director — orchestrates LLM call to generate effect plans."""

from __future__ import annotations

import sys

from beatlab.ai.plan import EffectPlan, parse_effect_plan, validate_effect_plan
from beatlab.ai.prompt import build_system_prompt, build_user_prompt
from beatlab.ai.provider import LLMProvider


def create_effect_plan(
    beat_map: dict,
    provider: LLMProvider,
    user_prompt: str | None = None,
) -> EffectPlan:
    """Send section data to LLM and get back a validated effect plan.

    Args:
        beat_map: Parsed beat map dict with sections.
        provider: LLM provider to call.
        user_prompt: Optional freeform creative direction from user.

    Returns:
        Validated EffectPlan.

    Raises:
        ValueError: If LLM response is not valid JSON or missing sections.
        Exception: If the API call fails.
    """
    system = build_system_prompt()
    user = build_user_prompt(beat_map, user_prompt=user_prompt)

    response_text = provider.complete(system, user)

    plan = parse_effect_plan(response_text)

    warnings = validate_effect_plan(plan)
    for w in warnings:
        print(f"  Warning: {w}", file=sys.stderr)

    return plan
