"""Google AI render pipeline — Nano Banana (stylize) + Veo (video between stills). No GPU needed."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from beatlab.render.google_video import GoogleVideoClient


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def _expand_sections_with_splits(
    sections: list[dict],
    plan_map: dict,
    splits_path,
    video_fps: float,
    default_style: str,
) -> tuple[list[dict], dict]:
    """Expand long sections into sub-sections using splits.json.

    Each section gets a `_file_key` field used for file naming:
    - Original sections: "042" (original index)
    - Sub-sections: "042_001", "042_002" (original index + sub index)

    This preserves cached files from pre-split runs (styled_042.png still valid)
    while adding new files for sub-sections (styled_042_001.png).

    Returns (expanded_sections, expanded_plan_map).
    """
    from beatlab.render.section_splitter import load_splits

    splits = load_splits(str(splits_path))
    split_map = splits.get("splits", {})

    if not split_map:
        # Tag all sections with their original file key
        for i, sec in enumerate(sections):
            sec["_file_key"] = f"{i:03d}"
        return sections, plan_map

    expanded = []
    expanded_plan = {}

    for i, sec in enumerate(sections):
        idx_str = str(i)
        if idx_str in split_map:
            sub_sections = split_map[idx_str]["sub_sections"]
            for j, sub in enumerate(sub_sections):
                new_sec = dict(sec)
                new_sec["start_time"] = sub["start_time"]
                new_sec["end_time"] = sub["end_time"]
                new_sec["start_frame"] = round(sub["start_time"] * video_fps)
                new_sec["end_frame"] = round(sub["end_time"] * video_fps)
                new_sec["_original_index"] = i
                new_sec["_sub_index"] = j
                new_sec["_file_key"] = f"{i:03d}_{j:03d}"

                new_idx = len(expanded)
                expanded.append(new_sec)

                # Inherit plan from parent with variation
                parent_plan = plan_map.get(i)
                if parent_plan:
                    from dataclasses import replace
                    sub_plan = replace(parent_plan, section_index=new_idx)
                    if sub.get("style_prompt"):
                        sub_plan.style_prompt = sub["style_prompt"]
                    elif j > 0 and sub_plan.style_prompt:
                        sub_plan.style_prompt = f"{sub_plan.style_prompt}, continuation with subtle evolution"
                    if j > 0 and sub_plan.transition_action:
                        sub_plan.transition_action = f"seamless continuation within {sec.get('label', 'section')}"
                    expanded_plan[new_idx] = sub_plan
        else:
            new_sec = dict(sec)
            new_sec["_original_index"] = i
            new_sec["_file_key"] = f"{i:03d}"

            new_idx = len(expanded)
            expanded.append(new_sec)

            if i in plan_map:
                from dataclasses import replace
                expanded_plan[new_idx] = replace(plan_map[i], section_index=new_idx)

    _log(f"  Split {len(split_map)} long sections → {len(expanded)} total (was {len(sections)})")
    return expanded, expanded_plan


def render_google_pipeline(
    video_file: str,
    beat_map: dict,
    effect_plan: object | None,
    work_dir: str,
    fps: float | None = None,
    default_style: str = "artistic stylized",
    progress_callback: Callable[[str, int, int], None] | None = None,
    vertex: bool = False,
    audio_descriptions: list[str] | None = None,
    motion_prompt: str | None = None,
    labels: bool = False,
    candidates: int = 0,
    backfill_candidates: bool = False,
    segment_filter: set[int] | None = None,
    intra_transition_prompt: str | None = None,
    ai_transitions: bool = False,
) -> str:
    """Run the full Nano Banana + Veo pipeline.

    Phase 1: Extract keyframes (one per section) from source video
    Phase 2: Nano Banana stylizes each keyframe
    Phase 3: Veo generates video transitions between consecutive styled keyframes
    Phase 4: Concatenate all transition clips, mux audio

    Each Veo clip morphs from styled_keyframe[i] → styled_keyframe[i+1],
    so every clip boundary is seamless — no stitching mismatch.

    Returns:
        Path to final assembled video.
    """
    work = Path(work_dir)
    frames_dir = work / "frames"
    styled_dir = work / "google_styled"
    segments_dir = work / "google_segments"
    output_path = work / "google_output.mp4"

    styled_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    sections = beat_map.get("sections", [])
    if not sections:
        raise ValueError("Beat map has no sections — Google pipeline requires sections")

    video_fps = fps or beat_map.get("fps", 30.0)

    client = GoogleVideoClient(vertex=vertex)

    # Build plan map
    plan_map: dict[int, object] = {}
    if effect_plan is not None:
        for sp in effect_plan.sections:
            plan_map[sp.section_index] = sp

    # ── Check for section splits (long sections broken into sub-sections) ──
    splits_path = work / "splits.json"
    if splits_path.exists():
        _log("Loading section splits...")
        sections, plan_map = _expand_sections_with_splits(
            sections, plan_map, splits_path, video_fps, default_style,
        )
        _log(f"  Expanded to {len(sections)} sections (from splits)")

    total_sections = len(sections)

    # Build file key list — used for all file naming
    file_keys = [sec.get("_file_key", f"{i:03d}") for i, sec in enumerate(sections)]

    # Copy parent styled images to first sub-section (sub_index 0)
    import shutil as _shutil
    for i, sec in enumerate(sections):
        if sec.get("_sub_index") == 0:
            parent_idx = sec.get("_original_index")
            parent_styled = styled_dir / f"styled_{parent_idx:03d}.png"
            sub_styled = styled_dir / f"styled_{file_keys[i]}.png"
            if parent_styled.exists() and not sub_styled.exists():
                _shutil.copy2(str(parent_styled), str(sub_styled))

    # ── Phase 1: Pick a keyframe per section ──
    _log(f"Phase 1: Selecting {total_sections} keyframes...")
    keyframe_paths: list[str] = []
    for i, sec in enumerate(sections):
        start_frame = sec.get("start_frame", int(sec["start_time"] * video_fps))
        end_frame = sec.get("end_frame", int(sec["end_time"] * video_fps))
        # Pick frame 1/3 into the section
        mid_frame = start_frame + (end_frame - start_frame) // 3
        kf_path = str(frames_dir / f"frame_{mid_frame:06d}.png")
        if not Path(kf_path).exists():
            kf_path = str(frames_dir / f"frame_{start_frame:06d}.png")
        keyframe_paths.append(kf_path)

    # ── Phase 2: Nano Banana stylization (with optional candidates) ──
    if candidates > 1:
        _log(f"Phase 2: Generating {candidates} candidates per section ({total_sections} sections, {total_sections * candidates} total)...")
    else:
        _log(f"Phase 2: Stylizing {total_sections} keyframes with Nano Banana...")

    from beatlab.render.candidates import generate_image_candidates, make_contact_sheet

    styled_paths: list[str] = []
    needs_selection: list[int] = []  # sections that have candidates but no selection yet

    for i, (sec, kf_path) in enumerate(zip(sections, keyframe_paths)):
        sp = plan_map.get(i)
        style = (sp.style_prompt if sp and sp.style_prompt else default_style)
        fk = file_keys[i]

        styled_path = str(styled_dir / f"styled_{fk}.png")

        if candidates > 1:
            cand_dir = work / "candidates" / f"section_{fk}"
            grid_path = str(work / "candidates" / f"section_{fk}_grid.png")

            # Already selected (styled image exists + candidates exist)
            if Path(styled_path).exists() and cand_dir.exists():
                _log(f"  [{i+1}/{total_sections}] Section {fk} (selected)")
                styled_paths.append(styled_path)
                continue

            # Styled image exists, no candidates
            if Path(styled_path).exists() and not backfill_candidates:
                _log(f"  [{i+1}/{total_sections}] Section {fk} (cached)")
                styled_paths.append(styled_path)
                continue

            # Backfill: promote existing styled to v1, generate v2-v4
            if Path(styled_path).exists() and backfill_candidates and not cand_dir.exists():
                import shutil as _shutil2
                cand_dir.mkdir(parents=True, exist_ok=True)
                _shutil2.copy2(styled_path, str(cand_dir / "v1.png"))
                _log(f"  [{i+1}/{total_sections}] Section {fk}: backfilling — existing → v1, generating v2-v{candidates}...")

            # Candidates generated but not yet selected
            if cand_dir.exists() and len(list(cand_dir.glob("v*.png"))) >= candidates:
                _log(f"  [{i+1}/{total_sections}] Section {fk} candidates (cached, awaiting selection)")
                needs_selection.append(i)
                styled_paths.append(styled_path)  # placeholder
                continue

            _log(f"  [{i+1}/{total_sections}] Section {fk}: generating {candidates} candidates...")

            def _stylize(source_path, style_prompt, output_path):
                try:
                    client.stylize_image(source_path, style_prompt, output_path)
                except Exception:
                    safe = f"abstract artistic interpretation, {sec.get('label', 'cinematic')}, dramatic lighting, surreal dreamlike atmosphere"
                    client.stylize_image(source_path, safe, output_path)
                return output_path

            paths = generate_image_candidates(
                section_idx=fk,
                source_image_path=kf_path,
                style_prompt=style,
                count=candidates,
                work_dir=str(work),
                stylize_fn=_stylize,
            )
            make_contact_sheet(paths, grid_path, i)
            _log(f"    Contact sheet → candidates/section_{fk}_grid.png")
            needs_selection.append(i)
            styled_paths.append(styled_path)  # placeholder
        else:
            # No candidates — single stylization
            if Path(styled_path).exists():
                _log(f"  [{i+1}/{total_sections}] Section {fk} (cached)")
                styled_paths.append(styled_path)
                continue

            _log(f"  [{i+1}/{total_sections}] Section {fk}: {style[:60]}...")
            try:
                client.stylize_image(kf_path, style, styled_path)
            except Exception as e:
                _log(f"  [{i+1}/{total_sections}] Content filter hit, retrying with safe prompt...")
                safe_style = f"abstract artistic interpretation, {sec.get('label', 'cinematic')}, dramatic lighting, surreal dreamlike atmosphere"
                try:
                    client.stylize_image(kf_path, safe_style, styled_path)
                    _log(f"  [{i+1}/{total_sections}] Retry succeeded with safe prompt")
                except Exception as e2:
                    _log(f"  [{i+1}/{total_sections}] FAILED even with safe prompt: {e2}")
                    raise

            styled_paths.append(styled_path)

        if progress_callback:
            progress_callback("stylize", i + 1, total_sections)

    # ── Phase 2.5: Selection gate — pause if candidates need selection ──
    if needs_selection:
        # Check which sections still need styled images (no selection applied yet)
        unselected = [i for i in needs_selection if not Path(styled_paths[i]).exists()]
        if unselected:
            fk_list = [file_keys[i] for i in unselected]
            _log(f"\n  ⏸  {len(unselected)} sections need candidate selection before Veo can proceed.")
            _log(f"  Review contact sheets in: {work}/candidates/")
            _log(f"  Sections: {fk_list}")
            video_name = work.name
            select_args = " ".join(f"{file_keys[i]}:v1" for i in unselected[:5])
            _log(f"  Run: beatlab select {video_name} {select_args} ...")
            _log(f"  Then re-run this render command to continue.\n")
            return str(output_path)  # Exit early — user needs to select

    # ── Phase 3: Veo segments between consecutive styled keyframes ──
    # Expand styled_paths for sections with sequence manifests
    import json as _json2
    expanded_styled: list[str] = []
    expanded_keys: list[str] = []
    expanded_section_idx: list[int] = []  # maps expanded index → original section index
    for i, sp in enumerate(styled_paths):
        manifest_path = styled_dir / f"styled_{file_keys[i]}_sequence.json"
        if manifest_path.exists():
            manifest = _json2.loads(manifest_path.read_text())
            for j, img_name in enumerate(manifest["images"]):
                expanded_styled.append(str(styled_dir / img_name))
                expanded_keys.append(f"{file_keys[i]}_seq{j}")
                expanded_section_idx.append(i)
        else:
            expanded_styled.append(sp)
            expanded_keys.append(file_keys[i])
            expanded_section_idx.append(i)

    # ── Phase 2.75: AI transition descriptions for intra-section pairs ──
    ai_intra_prompts: dict[int, str] = {}  # segment index → claude-generated prompt
    if ai_transitions and not intra_transition_prompt:
        _log("Phase 2.75: Claude describing intra-section transitions...")
        from beatlab.render.transition_describer import describe_transitions_batch

        # Collect intra-section pairs that need generation
        intra_pairs = []
        intra_indices = []
        num_segs = len(expanded_styled) - 1
        for i in range(num_segs):
            seg_path = str(segments_dir / f"segment_{expanded_keys[i]}_{expanded_keys[i+1]}.mp4")
            if segment_filter is not None and i not in segment_filter:
                continue
            if Path(seg_path).exists():
                continue

            orig_a = expanded_section_idx[i]
            orig_b = expanded_section_idx[i + 1]
            sec_a = sections[min(orig_a, len(sections) - 1)]
            sec_b = sections[min(orig_b, len(sections) - 1)]
            is_intra = (
                sec_a.get("_original_index") is not None
                and sec_a.get("_original_index") == sec_b.get("_original_index")
            )
            if is_intra:
                intra_pairs.append((expanded_styled[i], expanded_styled[i + 1]))
                intra_indices.append(i)

        if intra_pairs:
            _log(f"  {len(intra_pairs)} intra-section transitions to describe...")
            sp_styles = []
            for idx in intra_indices:
                orig = expanded_section_idx[idx]
                sp = plan_map.get(orig)
                sp_styles.append(sp.style_prompt if sp and sp.style_prompt else default_style)

            descriptions = describe_transitions_batch(
                intra_pairs, style_contexts=sp_styles, motion_prompt=motion_prompt or "",
            )
            for idx, desc in zip(intra_indices, descriptions):
                ai_intra_prompts[idx] = desc
            _log(f"  {len(ai_intra_prompts)} transitions described by Claude")

    # Use expanded lists for Veo generation
    num_segments = len(expanded_styled) - 1
    _log(f"Phase 3: Generating {num_segments} video segments with Veo (still→still)...")

    # Pre-compute all segment info: path, prompt, skip status
    segment_jobs: list[dict] = []
    for i in range(num_segments):
        seg_path = str(segments_dir / f"segment_{expanded_keys[i]}_{expanded_keys[i+1]}.mp4")

        # Skip if segment_filter is set and this segment isn't in it
        if segment_filter is not None and i not in segment_filter:
            segment_jobs.append({"index": i, "path": seg_path, "skip": "filter"})
            continue

        if Path(seg_path).exists():
            segment_jobs.append({"index": i, "path": seg_path, "skip": "cached"})
            continue

        # Map expanded indices back to original section indices
        orig_a = expanded_section_idx[i]
        orig_b = expanded_section_idx[i + 1]

        sp_a = plan_map.get(orig_a)
        sp_b = plan_map.get(orig_b)
        style_a = (sp_a.style_prompt if sp_a and sp_a.style_prompt else default_style)
        style_b = (sp_b.style_prompt if sp_b and sp_b.style_prompt else default_style)

        sec_a = sections[min(orig_a, len(sections) - 1)]
        sec_b = sections[min(orig_b, len(sections) - 1)]
        label_a = sec_a.get("label", "")
        label_b = sec_b.get("label", "")

        is_intra_section = (
            sec_a.get("_original_index") is not None
            and sec_a.get("_original_index") == sec_b.get("_original_index")
        )

        if is_intra_section:
            if i in ai_intra_prompts:
                # Claude-generated transition based on actual image content
                prompt_parts = [ai_intra_prompts[i]]
            elif intra_transition_prompt:
                prompt_parts = [
                    f"Smooth continuous cinematic video. Same visual world and atmosphere throughout.",
                    f"Visual style: {style_a}.",
                    intra_transition_prompt,
                ]
            else:
                prompt_parts = [
                    f"Smooth continuous cinematic video. Same visual world and atmosphere throughout.",
                    f"Visual style: {style_a}.",
                    f"Seamless fluid motion — no scene changes, no cuts, no dramatic transformations.",
                    f"The camera drifts slowly through the environment, revealing new angles and details of the same space.",
                ]
        else:
            action = (sp_b.transition_action if sp_b and sp_b.transition_action else None)
            if action:
                prompt_parts = [f"Cinematic video: {action}"]
                prompt_parts.append(f"Starting visual: {style_a}. Ending visual: {style_b}.")
            else:
                prompt_parts = [f"Cinematic video transitioning from {style_a} ({label_a}) into {style_b} ({label_b})."]

        if motion_prompt:
            prompt_parts.append(f"Camera and motion: {motion_prompt}.")

        if audio_descriptions:
            desc_a = audio_descriptions[i] if i < len(audio_descriptions) else ""
            desc_b = audio_descriptions[i + 1] if i + 1 < len(audio_descriptions) else ""
            if desc_a:
                prompt_parts.append(f"The music starts with: {desc_a[:200]}")
            if desc_b:
                prompt_parts.append(f"And transitions into: {desc_b[:200]}")

        prompt_parts.append("CRITICAL: The first frame of the video MUST be pixel-identical to the provided start image. The last frame MUST be pixel-identical to the provided end image. Do not alter, crop, zoom, or reinterpret the start and end frames in any way. Only generate motion and transformation for the frames in between.")

        segment_jobs.append({
            "index": i,
            "path": seg_path,
            "skip": None,
            "prompt": " ".join(prompt_parts),
            "start_img": expanded_styled[i],
            "end_img": expanded_styled[i + 1],
            "label": f"{expanded_keys[i]}→{expanded_keys[i+1]}: {label_a}→{label_b}",
            "intra": is_intra_section,
        })

    # Log skipped/cached segments
    to_generate = [j for j in segment_jobs if j["skip"] is None]
    cached = [j for j in segment_jobs if j["skip"] == "cached"]
    filtered = [j for j in segment_jobs if j["skip"] == "filter"]
    _log(f"  {len(to_generate)} to generate, {len(cached)} cached, {len(filtered)} filtered out")

    # Generate segments in parallel
    import concurrent.futures
    import threading

    max_workers = 10
    completed = 0
    lock = threading.Lock()

    def _generate_segment(job):
        nonlocal completed
        i = job["index"]
        intra_tag = " [smooth]" if job["intra"] else ""
        _log(f"  [{i+1}/{num_segments}] Segment {i}→{i+1} ({job['label']}) (8s){intra_tag}")
        try:
            client.generate_video_transition(
                job["start_img"], job["end_img"], job["prompt"], job["path"],
                duration_seconds=8,
            )
        except Exception as e:
            _log(f"  [{i+1}/{num_segments}] FAILED: {e}")
            raise
        with lock:
            completed += 1
            _log(f"  [{i+1}/{num_segments}] Done ({completed}/{len(to_generate)} complete)")
        if progress_callback:
            progress_callback("veo", completed, len(to_generate))

    if to_generate:
        if len(to_generate) == 1:
            _generate_segment(to_generate[0])
        else:
            _log(f"  Parallelizing with {min(max_workers, len(to_generate))} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_generate_segment, job): job for job in to_generate}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        _log(f"  Segment generation failed: {e}")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        raise

    # Build ordered segment_paths list
    segment_paths: list[str] = [j["path"] for j in segment_jobs]

    # ── Phase 3.5: Time-remap segments to match actual section durations ──
    _log("Phase 3.5: Time-remapping segments to match section durations...")
    remapped_dir = work / "google_remapped"
    remapped_dir.mkdir(parents=True, exist_ok=True)
    remapped_paths: list[str] = []

    for i in range(num_segments):
        # Target duration — for sequence sub-clips, split evenly within the section
        orig_a = expanded_section_idx[i]
        orig_b = expanded_section_idx[i + 1]
        sec_a_start = sections[min(orig_a, len(sections) - 1)].get("start_time", 0)
        sec_b_start = sections[min(orig_b, len(sections) - 1)].get("start_time", 0)
        target_duration = sec_b_start - sec_a_start

        # If both expanded entries map to the same section (sequence within one section),
        # split the section duration evenly among the sequence clips
        if orig_a == orig_b:
            sec = sections[min(orig_a, len(sections) - 1)]
            sec_dur = sec.get("end_time", 0) - sec.get("start_time", 0)
            # Count how many expanded entries belong to this section
            count = sum(1 for idx in expanded_section_idx if idx == orig_a)
            target_duration = sec_dur / max(1, count)

        if target_duration <= 0:
            target_duration = 8.0  # fallback

        remapped_path = str(remapped_dir / f"remapped_{expanded_keys[i]}.mp4")

        if Path(remapped_path).exists():
            remapped_paths.append(remapped_path)
            continue

        # Get actual duration of Veo clip
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", segment_paths[i]],
            capture_output=True, text=True,
        )
        try:
            actual_duration = float(probe.stdout.strip())
        except ValueError:
            actual_duration = 8.0

        speed_factor = actual_duration / target_duration

        if abs(speed_factor - 1.0) < 0.05:
            # Close enough — no remap needed
            import shutil
            shutil.copy2(segment_paths[i], remapped_path)
        else:
            _log(f"  Segment {i}: {actual_duration:.1f}s → {target_duration:.1f}s ({speed_factor:.2f}x)")
            # Use setpts for video speed, atempo for audio (if any)
            subprocess.run(
                ["ffmpeg", "-y", "-i", segment_paths[i],
                 "-filter:v", f"setpts={1.0/speed_factor:.4f}*PTS",
                 "-an",  # drop audio from Veo clips — we mux original audio later
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 remapped_path],
                capture_output=True, check=True,
            )

        remapped_paths.append(remapped_path)

    # ── Phase 3.75: Burn section labels (optional) ──
    if labels:
        _log("Phase 3.75: Burning section labels...")
        from beatlab.render.crossfade import burn_section_labels
        labeled_dir = str(work / "google_labeled")
        section_indices = list(range(num_segments))  # segment i = section i→i+1
        remapped_paths = burn_section_labels(remapped_paths, section_indices, labeled_dir)

    # ── Phase 4: Concatenate with crossfade and mux audio ──
    concat_output = str(work / "google_concat.mp4")
    if Path(concat_output).exists():
        _log("Phase 4: Concat cached, skipping...")
    else:
        _log("Phase 4: Assembling with 8-frame crossfades...")
        from beatlab.render.crossfade import concat_with_crossfade
        concat_with_crossfade(remapped_paths, concat_output, crossfade_frames=8, fps=video_fps)

    # Mux audio from original video
    muxed_output = str(work / "google_muxed.mp4")
    if Path(muxed_output).exists():
        _log("Phase 4.5: Mux cached, skipping...")
    else:
        _log("Phase 4.5: Muxing audio...")
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", concat_output,
             "-i", video_file,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-shortest",
             muxed_output],
            check=True, capture_output=True,
        )

    # ── Phase 5: Apply beat-synced effects (single-pass ffmpeg) ──
    _log("Phase 5: Applying beat-synced effects (single-pass ffmpeg)...")
    from beatlab.render.effects_ffmpeg import apply_effects

    apply_effects(
        video_path=muxed_output,
        output_path=str(output_path),
        beat_map=beat_map,
        effect_plan=effect_plan,
        fps=video_fps,
    )

    _log(f"Done! Output: {output_path}")
    return str(output_path)
