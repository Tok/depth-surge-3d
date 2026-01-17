# Codex Review Findings (Coding Guide Compliance)

Reviewed: 2026-01-17

## Scope
- Coding standards in `docs/CODING_GUIDE.md`
- Core orchestration (`StereoProjector`, `VideoProcessor`)
- Web UI orchestration and state handling (`app.py`)
- Utility modules (`utils/`, `processing/io_operations.py`)

## Findings

### High: Type hints still rely on legacy `typing` generics
**Location:** `src/depth_surge_3d/core/stereo_projector.py`,
`src/depth_surge_3d/core/constants.py`,
`src/depth_surge_3d/utils/progress.py`,
`src/depth_surge_3d/utils/resolution.py`,
`src/depth_surge_3d/models/video_depth_estimator.py`,
`src/depth_surge_3d/models/video_depth_estimator_da3.py`,
`src/depth_surge_3d/utils/video_processing.py`,
`src/depth_surge_3d/utils/batch_analysis.py`

**What:** Many modules still import `Optional`, `Dict`, `List`, `Tuple` from
`typing` instead of using built‑in generics (`list`, `dict`, `tuple`) and
`X | None`. This violates the “modern type hints” requirement.

**Why it matters:** The guide treats this as strict; it’s widespread and
undermines consistency.

**Suggested fix:** Add `from __future__ import annotations` where needed and
migrate annotations to built‑ins (e.g., `dict[str, Any]`, `str | None`).

---

### High: Invalid type defaults (`str = None`) in public APIs
**Location:** `src/depth_surge_3d/core/stereo_projector.py`

**What:** Several parameters are typed as `str` or `bool` but default to `None`
(e.g., `vr_format: str = None`, `baseline: float = None`). This conflicts with
the type hints and the guide’s strict typing requirement.

**Why it matters:** Type checkers will flag these, and it violates the strict
annotation policy.

**Suggested fix:** Use `str | None`, `float | None`, `bool | None`, or default
to concrete values and remove `None`.

---

### Medium: Global mutable state remains in the web UI
**Location:** `app.py`

**What:** `current_processing`, `ACTIVE_PROCESSES`, `SHUTDOWN_FLAG` are global,
mutated across threads and callbacks. This is still a major mutability hotspot.

**Why it matters:** Harder to test, race‑prone, and violates “immutable by
default.”

**Suggested fix:** Introduce a small state manager with an immutable dataclass
model (update via `replace()`), or confine mutation behind explicit methods.

---

### Medium: Duplicate progress logic across UI and utils
**Location:** `app.py`, `src/depth_surge_3d/utils/progress.py`

**What:** `app.py` maintains its own `ProgressCallback` with step tracking and
weighting, while `utils/progress.py` provides similar functionality.

**Why it matters:** Violates DRY and makes behavior diverge between CLI and UI.

**Suggested fix:** Reuse `utils.progress.ProgressCallback` in `app.py`, or move
the UI-specific behavior into a thin adapter.

