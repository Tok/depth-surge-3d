# Codex Review Findings — v0.9.0 Pre‑Release

Date: 2026-01-18  
Version: 0.9.0 (pre‑release)

## Scope
- Real-time preview system (`app.py`, `templates/index.html`)
- AI upscaling (`src/depth_surge_3d/models/upscaler.py`)
- 8‑step pipeline + upscaling integration (`video_processor.py`)
- Path handling & file management (`app.py`)
- CI/config checks (`.github/workflows/ci.yml`)

## Findings (Ordered by Severity)

### High: Unverified model downloads + `torch.load` on remote weights
**Location:** `src/depth_surge_3d/models/upscaler.py`

**What:** Real‑ESRGAN weights are downloaded at runtime via `urllib.request.urlretrieve`
and loaded with `torch.load` without checksum verification or signature validation.
This opens a supply‑chain risk (malicious model file) and uses `torch.load` on
untrusted input.

**Why it matters:** `torch.load` can execute arbitrary code if a compromised
file is loaded. This is a high‑impact security risk.

**Suggested fix:** Pin SHA‑256 hashes for each model and verify after download,
or ship checksums in constants. Prefer `torch.load(..., weights_only=True)` if
supported, and add an explicit download timeout.

```203:246:src/depth_surge_3d/models/upscaler.py
            if not model_path.exists():
                urllib.request.urlretrieve(model_url, model_path)
...
            loadnet = torch.load(model_path, map_location=self.device)
```

---

### High: Arbitrary output directory accepted from client
**Location:** `app.py`

**What:** `/process` accepts `output_dir` from the request, resolves it, and
uses it directly. There is no guard to ensure the path is under the configured
`OUTPUT_FOLDER`, which allows a client to point to any path on disk.

**Why it matters:** This enables processing arbitrary files on the host if the
server is exposed beyond localhost. It’s also a path traversal risk.

**Suggested fix:** Enforce that `output_dir` is within
`Path(app.config["OUTPUT_FOLDER"]).resolve()`, and reject any path outside.

```905:949:app.py
    output_dir = Path(output_dir_str).resolve()
    if not output_dir.exists():
        return jsonify({"error": f"Output directory not found: {output_dir}"}), 404
    video_path = find_source_video(output_dir)
```

---

### Medium: Preview frames have no size cap or validation
**Location:** `app.py:ProgressCallback.send_preview_frame`

**What:** Preview encoding downscales to a fixed width, but does not guard
against zero‑width images, oversize inputs, or cap base64 payload size. Errors
are swallowed silently, making failures opaque.

**Why it matters:** Large frames can still produce big payloads (memory/latency)
and the silent failure makes debugging preview issues difficult.

**Suggested fix:** Validate image dimensions, guard width==0, add a max byte
limit for base64 payloads, and log exceptions with context.

```309:364:app.py
            frame = cv2.imread(str(frame_path))
            ...
            _, buffer = cv2.imencode(".png", frame_small)
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            socketio.emit("frame_preview", preview_data, room=self.session_id)
```

---

### Medium: Upscale preview writes intermediates even when disabled
**Location:** `src/depth_surge_3d/processing/video_processor.py`

**What:** When `keep_intermediates=False`, `_process_upscaling_frames` still
creates `left_upscaled/right_upscaled` directories and writes preview frames.
Those temp files are never cleaned up.

**Why it matters:** Disk usage grows unexpectedly in “no intermediates” mode.

**Suggested fix:** Write previews to an explicit temp directory and delete after
use, or send previews from memory without writing files.

```1540:1618:src/depth_surge_3d/processing/video_processor.py
        if not left_upscaled:
            left_upscaled = directories["base"] / INTERMEDIATE_DIRS["left_upscaled"]
        ...
        if settings["keep_intermediates"] or should_send_preview:
            cv2.imwrite(str(left_upscaled_path), left_upscaled_img)
```

---

### Medium: Coding guide compliance — legacy typing remains
**Location:** multiple files (`core/constants.py`, `core/stereo_projector.py`,
`utils/progress.py`, `utils/resolution.py`, `models/*`, `utils/batch_analysis.py`)

**What:** Several modules still import `Optional`, `Dict`, `List`, `Tuple` from
`typing` instead of using built‑in generics (`dict`, `list`, `tuple`) and
`X | None`.

**Why it matters:** This violates the “modern type hints” requirement in
`docs/CODING_GUIDE.md`.

**Suggested fix:** Add `from __future__ import annotations` where needed and
convert annotations to built‑ins.

---

### Low: CI does not enforce coverage threshold
**Location:** `.github/workflows/ci.yml`

**What:** Tests run with coverage reporting but do not enforce a minimum
threshold (`--cov-fail-under` is missing).

**Why it matters:** The release target is 90% coverage, but CI won’t fail when
coverage drops.

**Suggested fix:** Add `--cov-fail-under=90` (or the current target) to unit test
coverage command.

```45:48:.github/workflows/ci.yml
pytest tests/unit -v --cov=src/depth_surge_3d --cov-report=xml --cov-report=term
```

