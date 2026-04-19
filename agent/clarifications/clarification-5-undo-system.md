# Clarification 5: Undo System

**Purpose**: Clarify requirements for implementing undo/redo functionality for timeline editing operations  
**Created**: 2026-04-09  
**Status**: Awaiting Responses  
**Source Context**: Chat conversation — undo system design discussion  

---

## How to Use This Document

Answer questions by typing responses after `>` markers. Leave feedback or follow-up questions in HTML comment blocks (`<!-- your feedback -->`). Update Status to "Completed" when done. Run `@acp.clarification-address` to have the agent analyze your responses.

---

# Item 1: Approach Selection

## Questions 1.1: Implementation Strategy

- Do you prefer the SQLite trigger approach (automatic, captures all DB changes, no per-endpoint work)?

>  y

- Or do you prefer the command pattern (explicit per-endpoint recording, can include file references)?

> both I believe

- A hybrid is also possible — triggers for DB + command entries for asset references. Acceptable?

>  hybrid yes

<!-- [Agent Analysis]
Confirmed: Hybrid approach.
- SQLite triggers on all tracked tables capture inverse SQL automatically into undo_log
- Command layer provides operation grouping (undo_group ID) and human-readable descriptions
- Undo execution: disable triggers → execute inverse SQL for group in reverse → re-enable triggers
-->

>

## Questions 1.2: Persistence

- Undo history should persist across server restarts (stored in project.db)?

> y

- Should undo history persist across sessions (i.e. user can undo operations from a previous day)?

> y

- Should there be a max undo history depth (e.g. 100 operations, or unlimited)?

> I think 1,000 should be reasonable, it's very unlikely the user will need
> to undo more than say 100 events, but useful to track them. 

<!-- [Agent Analysis]
1,000 undo groups. When exceeded, prune oldest entries: `DELETE FROM undo_log WHERE undo_group <= (SELECT MIN(undo_group) FROM undo_log)` until count is under limit. The undo_log rows themselves may be more numerous (multiple rows per group for multi-table operations), but the group count is what's capped.
-->

>

---

# Item 2: Scope

## Questions 2.1: What Operations Support Undo

- Keyframe updates (prompt, timestamp, source, selected candidate)?

> Split

<!-- [Agent Analysis]
Interpreted "Split" as: keyframe updates should be tracked at a split/granular level — each field change captured individually by the trigger, grouped into one undo step by the command layer. Confirmed.
-->

>

- Transition updates (action, remap, selected, opacity_curve, color grading curves)?

> Y

- Soft-deletes (keyframes, transitions)?

> Y

- Keyframe/transition creation (undo = soft-delete the new item)?

> Y

- Suppression zone changes?

> Y

- User effect changes?

> Y

- Batch operations (batch-set-base-image, bulk pool insert)?

>  Y

- Track operations (add/remove tracks, reorder)?

> Y

## Questions 2.2: What Should NOT Support Undo

- Video/image generation (Veo, Replicate) — these are expensive and produce assets. Undo should not delete generated assets?

> correct

- Render operations — undo should not delete render outputs?

> correct

- Settings changes (settings.yaml) — should these be undoable?

> what settings exist?

<!-- [Agent — Researched]
Current settings in `settings.yaml`:
- `preview_quality: 50` — preview render quality
- `image_backend: replicate/nano-banana-2` — which service generates keyframe images
- `video_backend: runway/veo3.1_fast` — which service generates transition videos
- `crossfade_frames` — (optional) crossfade duration for renders

These are project-level config, not timeline edits. Changing `video_backend` from `vertex` to `runway` is a config choice, not a creative edit.

**Recommendation**: Exclude settings from undo — they're operational config, not creative timeline state.

Would you like to accept this recommendation? (yes/no)
-->

> y

---

# Item 3: Grouping and Granularity

## Questions 3.1: Operation Grouping

- Should related changes be grouped into one undo step? For example, "split transition" creates 2 new transitions + modifies 1 — undo should revert all 3 in one step?

> y

- Should the frontend's "save" action (which may batch multiple field changes) count as one undo step?

> where is save?

<!-- [Agent — Researched]
There is no explicit "save" button in the frontend. Changes are saved immediately via individual API calls:
- Editing a keyframe prompt → `POST /update-keyframe`
- Changing a transition's remap → `POST /update-transition-style`
- Moving a keyframe → `POST /update-keyframe` with new timestamp

Each API call is a separate operation. The frontend doesn't batch multiple field changes into one save — each field change triggers its own endpoint call. So this question is moot: each API call = one undo group naturally.

However, some frontend actions do trigger multiple API calls rapidly (e.g. drag-drop a keyframe updates timestamp + recomputes transitions). These would create multiple undo groups unless we add a debounce/coalesce window (e.g. group operations within 500ms into one undo step).
-->

> checkpoint is the closest thing to a save.

<!-- [Agent Analysis]
Checkpoints are full DB snapshots (`project.db.checkpoint-*` files) created via `POST /api/projects/:name/checkpoint`. They're coarse-grained named save states — orthogonal to the fine-grained undo_log.

**Relationship to undo**:
- Each API call = one undo_group (fine-grained, automatic)
- Each checkpoint = full DB snapshot (coarse-grained, explicit)
- Undo walks backward through individual operations
- Restoring a checkpoint wholesale REPLACES the current DB (all undo_log lost, or at least invalidated)

**Recommendation**: Treat checkpoints as independent from undo:
- Taking a checkpoint does NOT create an undo entry — it's a "save-as" not an edit
- Restoring a checkpoint should clear the undo_log (since the state it refers to no longer exists)
- The undo history panel could optionally show checkpoint markers inline so users know where "save points" are in the timeline of operations

Would you like to accept this recommendation? (yes/no)
-->

>

- When you split a transition, the undo needs to know the original video reference. Should we store a snapshot of the full row state before modification?

> we need to store all context required to perform an undo

<!-- [Agent Analysis]
The SQLite trigger approach handles this automatically — the UPDATE trigger captures all old column values in the inverse SQL statement. For a split:
1. Trigger on `UPDATE transitions SET to_kf=... WHERE id='tr_X'` → logs `UPDATE transitions SET to_kf='original_to_kf' WHERE id='tr_X'`
2. Trigger on `INSERT transitions` (new tr_Xb) → logs `DELETE FROM transitions WHERE id='tr_Xb'`
3. Trigger on `INSERT transitions` (new tr_Xc) → logs `DELETE FROM transitions WHERE id='tr_Xc'`

All grouped under one undo_group. Undo replays in reverse: delete the new transitions, restore the original `to_kf`. The video file stays on disk — the DB pointer (`selected`) is restored to the original value by the trigger.

No explicit snapshot needed — the triggers ARE the snapshot.
-->

> y

## Questions 3.2: Redo Support

- Do you need redo (undo the undo)?

> plan for it, but not P0

<!-- [Agent Analysis]
"Plan for it, not P0" interpretation — design the schema to support redo from day one, ship without exposing it:

**Schema (ship in P0)**:
- `undo_log` has `undone` boolean column (default 0)
- Instead of DELETE on undo, flip `undone = 1` so the row can be re-applied
- `redo` = find most recently-undone group, re-apply its original SQL (not the inverse)
- This means storing both forward and inverse SQL per undo_log row

**Frontend (ship later)**:
- Ctrl+Shift+Z binding
- Redo button in undo history panel
- `POST /api/projects/:name/redo` endpoint

**Tradeoff**: Storing both forward+inverse SQL roughly doubles undo_log size, but SQLite rows are cheap and 1000 groups * ~10 rows each = 10k rows max. Negligible.

Would you like to accept this recommendation? (yes/no)
-->

>

- Should redo history be cleared when a new operation is performed (standard behavior)?

> y

---

# Item 4: Interface

## Questions 4.1: API

- `POST /api/projects/:name/undo` — undo last operation. Acceptable?

>  y

- `POST /api/projects/:name/redo` — redo last undo. Acceptable?

>  y

- `GET /api/projects/:name/undo-history` — list recent operations. Acceptable?

>  y

## Questions 4.2: CLI

- `.venv/bin/python3 -m beatlab undo` — undo last operation from CLI?

>  y

- `.venv/bin/python3 -m beatlab undo-history` — show recent history from CLI?

>  y

## Questions 4.3: Frontend Integration

- Should the frontend show an undo history panel?

>  y

- Keyboard shortcut (Ctrl+Z / Cmd+Z) for undo in the editor?

>  y

---

# Item 5: Edge Cases

## Questions 5.1: Conflict Handling

- If a transition was modified, then a video was generated for it, then user undoes the modification — the video may no longer match. Should undo warn about this?

> this is the correct behavior. it should reset to whatever the video selection
> was before the transition modification

<!-- [Agent Analysis]
Confirmed: undo restores the DB state (including `selected` pointer) to pre-modification. The generated video file stays on disk as a candidate — it just won't be selected anymore. No warning needed; this is the expected behavior.
-->

> y

- If undo would restore a deleted transition that now overlaps with a newly created one — should it fail, warn, or force?

> In theory, they would have to undo their tr/kf add before hitting this
> part of the stack, so it should be a non-issue

<!-- [Agent Analysis]
Correct — undo is a strict stack. To reach the "restore deleted tr" operation, the user must first undo the "create new tr" operation that caused the overlap. By the time they hit the restore, the overlapping transition is already gone. The stack ordering prevents this conflict by construction.
-->

> y

## Questions 5.2: Concurrent Access

- If multiple browser tabs are editing the same project, should undo be per-tab or global?

> undo is global, the user can only edit the project with one instance at a time.
> This is simply because it's the the way the rest of the app works. If the 
> user wants to work in parallel, they should duplicate the project or use
> more than one timeline.

---
