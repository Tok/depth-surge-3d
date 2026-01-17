# Codex Review Findings (Coding Guide Compliance)

Reviewed: 2026-01-17

## Scope
- Coding standards in `docs/CODING_GUIDE.md`
- Core orchestration (`StereoProjector`, `VideoProcessor`)
- Web UI orchestration and state handling (`app.py`)
- Utility modules (`utils/`)

## Findings

### High: Missing type hints across public functions
**Location:** `app.py`

**What:** Many public functions and methods lack type annotations despite the
guide requiring complete type hints on all functions. Examples: `get_video_info`,
`get_system_info`, `upload_video`, `start_processing`, `stop_processing`, and
`ProgressCallback.update_progress`.

**Why it matters:** Violates strict standards, reduces IDE/static checking, and
makes refactoring riskier.

**Suggested fix:** Add full annotations, using modern built-in generics
(`list[...]`, `dict[...]`, `str | None`).

---

### Medium: Global mutable state used as shared control plane
**Location:** `app.py`

**What:** `current_processing` is a global dict mutated across threads and
callbacks. This violates “immutable by default” and makes side effects hard to
reason about and test.

**Why it matters:** Data races, fragile state transitions, and harder unit tests.

**Suggested fix:** Introduce a `@dataclass(frozen=True)` state model, and update
via `replace()` through a dedicated state manager (or at least encapsulate
mutations behind methods).

---

### Medium: “Pure utils” module performs side effects
**Location:** `src/depth_surge_3d/utils/file_operations.py`

**What:** Module docstring claims “pure functions without side effects,” but
functions run `subprocess`, access the filesystem, and create directories.

**Why it matters:** Breaks the “pure functions live in utils” policy and makes
testing/mocking harder.

**Suggested fix:** Split into `utils/video_utils.py` (pure helpers) and
`processing/io.py` (side-effectful operations), and update imports.

---

### Medium: Overlong functions and mixed responsibilities
**Location:** `src/depth_surge_3d/processing/video_processor.py`,
`app.py`

**What:** Several functions exceed the 20-line limit and mix orchestration,
logging, progress updates, and data transformation. Examples:
`_generate_depth_maps_chunked`, `_determine_chunk_params`,
`ProgressCallback.update_progress`.

**Why it matters:** Increases complexity and violates the small/pure function
guideline.

**Suggested fix:** Extract pure helpers (e.g., chunk parameter selection,
progress calculation) and keep orchestration minimal.

---

### Medium: Magic numbers remain inline
**Location:** `src/depth_surge_3d/processing/video_processor.py`

**What:** Resolution thresholds and chunk sizes are inline literals
(e.g., `2160`, `1440`, `8.0`, `4`, `6`, `12`).

**Why it matters:** Violates “no magic numbers,” complicates testing and tuning.

**Suggested fix:** Move thresholds and chunk sizes into
`src/depth_surge_3d/core/constants.py` with descriptive names.

---

### Low: Legacy `typing` imports are still used
**Location:** `src/depth_surge_3d/processing/video_processor.py`,
`src/depth_surge_3d/utils/file_operations.py`

**What:** `List`, `Dict`, `Optional` are still used in places instead of modern
built-in generics.

**Why it matters:** Non-compliant with the guide and adds unnecessary verbosity.

**Suggested fix:** Replace with `list`, `dict`, `tuple`, `str | None`.

