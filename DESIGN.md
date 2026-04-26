# buddhabrot — design notes

Captured at a handoff point. Reflects actual code at commit-of-this-writing.
Scope: what's implemented, why certain choices were made, where the design is
rough, and what's worth iterating on next.

---

## 1. What this is

An interactive explorer + composer for a **4D buddhabrot** — the Mandelbrot
orbit-density visualization lifted from its classic 2D c-plane presentation
into the full 4D state space `(c_re, c_im, z_re, z_im)`, with a rotatable
SO(4) orientation picking the viewing plane.

Two modes:

- **Live** — a real-time sampler fills a histogram, palette-applied to pixels
  every frame. Drag to rotate, scroll to zoom, press to take keyframe
  screenshots.
- **Composer** — browse the saved screenshots, drop them onto a timeline,
  scrub a playhead with live interpolated-frame preview, then render a video
  through those keyframes via an ffmpeg subprocess.

CLI subcommands mirror the non-interactive operations: `render-frame`,
`render-video` (with `--resume`).

---

## 2. Module map

```
src/
  main.rs          argv dispatch → UI or CLI
  app.rs           UI state machine, event loop, all per-mode drawing
  live.rs          — (no: live-mode logic currently lives inline in app.rs)
  composer.rs      composer state, thumbnail loader, preview worker,
                   interpolation helper
  view.rs          View { SO(4) rotation, pan, zoom, projection }
  sampler.rs       SamplerHandle + worker_loop (used by both UI and CLI)
  render.rs        single-frame `render_frame(spec) -> RGBA` wrapper around
                   the sampler
  videorender.rs   video pipeline: per-frame PNGs → ffmpeg assembly;
                   Catmull-Rom interpolation helpers (pub'd for composer use)
  session.rs       `.render-<ts>/session.json` on-disk schema + scan
  timeline_save.rs composer-timeline persistence (autosave slot + manual
                   snapshots; key=value format like session.rs)
  metadata.rs      parse saved PNG's tEXt chunks back into a FrameSpec
  savepng.rs       write PNG with Buddhabrot.* tEXt metadata
  palette.rs       5 builtin palettes as 1024-entry LUTs
  onion.rs         drag-time onion-skin overlay
  overlay.rs       fontdue text + fill_rect
  gizmo.rs         tesseract orientation gizmo, face pick → corner-view snap
  orbit.rs         z²+c iteration + cardioid/bulb rejection
  input.rs         drag modifier → rotation plane mapping
  cli.rs           `render-frame` / `render-video` argv parsers
  session.rs       session directory schema
```

`app.rs` is large (~3,100 lines). Worth splitting but currently monolithic
and internally consistent via shared layout helpers.

---

## 3. What's built

### 3.1 Live mode

- **Fractal pipeline**: `SamplerHandle` spawns `available_parallelism() - 1`
  worker threads. Each owns an MH (Metropolis-Hastings) chain, shares one
  `Vec<AtomicU32>` histogram. Uniform warmup + MH chain per worker diversifies
  coverage. Each worker polls `view.generation` between batches for
  preemption (<5ms to react to a view change).
- **4D view**: `View.rotation: [[f64;4];4]` updated via Givens row-rotations
  with periodic Gram-Schmidt reorthonormalization. Pan is a 2D world-space
  offset, zoom is `half_width` in world units.
- **Drag controls** (6 rotation planes mapped to mouse × modifier):
  - left-drag → XZ / YW
  - shift+left-drag → XY (c-spin) / ZW (z-spin)
  - ctrl+left-drag → XW / YZ
  - right-drag → XZ / YZ (camera-style orbit, depth=Z; both axes inverted
    so cursor motion feels like grabbing the object)
  - shift+right-drag → XW / YW (orbit with depth=W)
  - middle drag → pan (y-inverted so drag feels like "grab")
  - scroll → zoom (mouse-anchored)
- **Undo/redo**: 256-entry snapshot stack (rotation + center + half_width +
  max_iter). Pushed on drag-end, scroll, snap-complete, reset, max-iter
  change. Coalesced within 350ms windows. `meta+z` / `meta+shift+z`.
- **Gizmo**: corner-bottom-right tesseract with 24 faces. Clicking a face
  snaps the orientation to a 45° "corner view" of that plane with the signs
  of the two fixed axes determining the tilt direction (→ 24 distinct snap
  orientations). Snap is a 250ms eased rotation animation.
- **Onion skin** (toggle `o`): captures a snapshot pre-drag and displays it
  under the sparse live frame during a drag, affine-mapped through the
  rotation delta. Provides spatial continuity as the rendering catches up.
- **Save frame button**: PNG → `~/buddhabrot/buddhabrot-<ts>.png` with
  `Buddhabrot.*` tEXt chunks encoding the exact view for later load. Appends
  to live-mode's keyframe list.
- **Render dialog**: width/height text fields, scale-% slider, samples-per-
  output-pixel-per-frame slider (log-scaled, 0.1–200), **test** button
  renders one frame on a background thread → ETA from measured time ×
  total-frames.
- **Render queue + status widget**: top-right sibling to the tab bar;
  progress bar + frame counter while rendering, "+N queued" when stacked,
  "last: /path.mp4" / "failed: …" when idle. Renders spawn detached
  subprocesses that survive UI exit; new UI instances adopt in-flight
  sessions by scanning `~/buddhabrot/.render-*/`.
- **Pause button** (`space`): idles all sampler workers so CPU frees for a
  background render.
- **Auto-pause** at `UI_AUTO_PAUSE_SAMPLES = 10B` proposals (~12.7k/pixel
  at 1024×768). Workers flip `paused=true` once and disarm by setting
  `auto_pause_at = u64::MAX`; `invalidate()` re-arms on the next view
  change *and* clears `paused` so any camera move resumes sampling fresh.
- **Palette / gamma HUD** (lower-left): a stacked block with the same
  controls as the composer's preview pane — gradient bar with editable
  stops (click bar to add, drag stop to move, click stop for an RGB
  popover, right-click stop to delete, ≥2 enforced), `reset` button (re-
  seeds the current preset's stops), and a gamma slider underneath. Uses
  the same widgets as `preview_pane_layout`'s palette section. State lives
  on `App` (`palette_stops`, `palette`, `palette_color_edit`,
  `live_widget_drag`); editing rebuilds the 1024-LUT synchronously and the
  next frame's `palette::apply` picks it up. The `p` / shift-`p` keys
  still cycle palette presets — they now load the next/previous entry of
  `palette::builtin_stops()` into the editable stops (replacing whatever
  the user had). The `g` / shift-`g` keys still nudge gamma directly.

### 3.2 Composer mode

- **Tabs** (centered top) switch Live ↔ Composer.
- **Thumbnail grid** (top-left): background thread decodes every PNG in
  `~/buddhabrot/`, box-downsampled to max 320px. Recent-first, scrollable.
  Thumb-size slider above. Click to toggle add/remove from the timeline.
  Right-click opens a 2-row context menu: **Use in Live** (loads the
  screenshot's view metadata into the live `View`, switches mode to Live,
  pushes undo snapshots) and **Trash** (deletes the file via `trash::`).
- **Timeline** (bottom): thumbnail strip with:
  - **drag-reorder** — press a thumb, ghost follows cursor, blue insertion
    indicator, release to commit.
  - **drag-the-gap** — press between two thumbs, drag to stretch/compress
    spacing. Weights persist to each `TimelineItem.spacing` and affect both
    preview and final video render.
  - three-band delta arrow in each gap: magenta = rotation magnitude, blue =
    pan, yellow = zoom.
- **Preview pane** (top-right):
  - Interpolated frame rendered at `preview_quality_pct` of pane size, 200k
    samples for sub-second turnaround.
  - Playhead slider (drag to scrub).
  - Play/pause button (toggles `composer.playing`; auto-advance loops).
  - `playhead • WxH • q% • Nms` status line.
  - Size + spp sliders.
  - **Palette editor**: gradient bar with click-to-add stops, drag to
    reposition, right-click marker to delete (≥2 stops enforced),
    left-click marker to open an RGB-slider popover (close by clicking
    outside the palette strip). `reset` button restores the default stops.
  - **Gamma slider** (under the palette stops): overrides per-keyframe
    gamma during preview, range `[0.1, 2.5]`.
  - Falls back to nearest keyframe's thumbnail while the first frame is still
    rendering.
- **Controls pane**: `cycle palette` override (None or one of builtins),
  `generate rotation keyframes` (auto-generates a PNG ring around each of the
  6 SO(4) planes from the current live view), `clear`, `render timeline` (→
  opens the same dialog as live mode, pre-populated with the timeline's
  keyframes + current spacings).
- **Embedded render panel** (bottom-right of the right column): width/height
  text fields (Tab cycles between them), `spp` slider, predicted **estimate**
  line, and a 2-column × 3-row resolution radio grid (4K / 1440p / 1080p /
  720p / 480p / 240p) sitting to the right of the fields. Clicking a radio
  fills both fields; typing a custom value naturally deselects all radios
  (active = exact match against `RESOLUTION_PRESETS`). Below: **clear** /
  **render** buttons and a queue list. The active-render row in the queue
  shows `▶ cur/total • eta hh:mm:ss • s.ss s/frame • <filename>` with the
  filename middle-truncated; the static `estimate:` line above stays as
  the *predictor* for the next render based on the test measurement.
- **Timeline persistence** (`timeline_save.rs`):
  - **Autosave slot**: `~/buddhabrot/timeline.state`, rewritten on every edit
    (timeline mutation, stepper click, gamma/spp slider, palette edit, etc.).
    Writes are rate-limited to one every 250ms via a `save_dirty` flag
    flushed from `Composer::tick`. Restored in `Composer::new` — relaunching
    the UI brings back the last session automatically.
  - **Manual snapshots**: `save` button writes a timestamped file to
    `~/buddhabrot/timelines/timeline-<ts>.state`; `load` button opens a
    dialog listing every `.state` in that directory (newest first, with
    keyframe-count badges). Autosave and manual saves live in different
    paths so a fast autosave can't clobber a named manual save.
  - **Format**: `key=value\n` lines, hand-parsed (no serde dep). Fields:
    fps, rp_width/height/spp, gamma, playhead, palette stops (pos + rgb),
    and per-item `source` / `segment` / `dwell`. Versioned (`version=1`).
  - **Missing-keyframe handling**: on load, each item is kept in the
    timeline regardless of whether its PNG still exists on disk. Items with
    a vanished source get `missing: true` and a placeholder `FrameSpec`;
    the timeline draws them dimmed with a red X and a "missing" caption,
    and `Composer::missing_paths()` gates render spawns (both the embedded
    panel button and the live-mode-style dialog). User fixes by restoring
    the file or removing the slot.
  - **Top-left strip**: `save` / `load` buttons live in their own band
    between the tab bar and the grid; the thumb-size slider shares the
    strip to its right. The load dialog anchors under the load button and
    flips upward near the bottom edge of the window.

### 3.3 Render pipeline (shared UI ↔ CLI)

1. A render is a `RenderJob` — keyframes, W×H, fps, samples, total_frames,
   output path, session_dir, optional spacings.
2. `spawn_render_job_free` forks `$exe render-video …`, detached
   (`process_group(0)`, null stdin, inherited stderr).
3. The subprocess creates `<session_dir>/`, writes `session.json` atomically,
   loops: render frame → save as `frames/NNNNN.png` → update json.
4. After every frame exists, invokes `ffmpeg -framerate … -i frames/%05d.png
   -c:v libx264 -pix_fmt yuv420p OUT`.
5. On success, deletes the session dir.
6. UI watcher thread polls `session.json` every 500ms (with a 15s startup
   grace before considering a missing dir as "failed"). Same mechanism serves
   live, composer, adopted, and resumed renders identically.
7. `--resume <dir>` re-reads `session.json`, skips frames already on disk,
   continues.

### 3.4 Interpolation

Between keyframes: Catmull-Rom on matrix entries (then Gram-Schmidt
reorthonormalized), pan, `log(half_width)` (linear-cr was tried — it
made zoom pacing feel uniform against pan/rotation, but Hermite on raw
`half_width` overshoots below zero whenever a zoomed-out keyframe sits
next to a much deeper one; the clamp rescued the renderer but produced
blank preview frames, so zoom is back in log space), `gamma`, `max_iter`.
Palette LUTs are linearly blended (Catmull-Rom on u8 colors overshoots).
Keyframe `half_width` is floored at `1e-12` before `ln()` so a degenerate
zero/negative input can't produce `-inf`.

`segment_for_playhead(n, weights, t)` resolves a playhead `t ∈ [0,1]` into
`(segment_idx, local_t)`. Uniform when `weights` is empty; otherwise
cumulative-weight mapping. Used identically by preview and video renderer.

---

## 4. Design intents

### 4.1 Keep the sampler the single source of truth

Everything that produces a buddhabrot frame — live UI, CLI render-frame,
video renderer, composer preview — goes through the same `sampler::worker_loop`
via `render::render_frame(FrameSpec)`. Preemption (`view.generation`) and
target-samples (`Option<u64>`) are the only two control axes. This is why
`FrameSpec` carries `n_workers: usize` as a field: callers dial parallelism
per purpose (UI sampler uses all cores-minus-one, preview uses a dedicated
lane via its own thread, CLI accepts `--threads`).

### 4.2 Render subprocesses are independent of the UI

The MP4 render is many minutes of CPU. Tying it to the UI's lifecycle would
mean a crash or accidental close loses work. So:

- Subprocess is detached (own process group) on spawn.
- Intra-run state lives on disk as `session.json` + per-frame PNGs.
- `println!` was replaced with fallible `writeln!` specifically because Rust
  panics on EPIPE when the UI's stdout pipe closes.
- UI doesn't own the render in any meaningful sense — it scans for sessions
  on startup, adopts live ones, offers to resume dead ones.

This has the nice property that the CLI (`render-video` standalone) and the
UI use exactly the same code path.

### 4.3 Save-PNG is both a screenshot and a seed for video

Every UI save captures enough metadata (`Buddhabrot.Rotation`, `Center`,
`HalfWidth`, `MaxIter`, `Palette`, `Gamma`) to fully reconstruct a `FrameSpec`.
This is what makes the composer work — its timeline items are just references
to PNGs, and interpolation pulls in their metadata lazily.

The metadata is stored as PNG tEXt chunks (not true EXIF — see limitations).
`exiftool` and `identify -verbose` both surface it.

### 4.4 Borrow-checker ergonomics

winit's `window_event` takes `&mut self`. We hold `Option<WindowState>` inside
`App`; grabbing `ws = self.ws.as_mut().unwrap()` for the whole function
locks `self.ws` mutably for the whole scope. Calling any `&mut self` method
afterwards conflicts.

Pattern adopted: anything that needs to be callable while `ws` is held is a
**free function** taking the specific Arcs/fields it touches
(`spawn_render_job_free`, `push_view_snapshot_fn`, `composer_handle_click`,
etc.). Methods on `App` are reserved for call sites that don't overlap with
`ws`. It's a small price for keeping the event loop in one function.

### 4.5 Dialogs as the canonical render entry point

Click "render" in live mode or "render timeline" in composer, you get the
same dialog. `RenderDialogState.keyframes` + `.spacings` are captured at
open, so the dialog is source-agnostic. This cost nothing but a few more
fields on the struct and unified a lot of code.

### 4.6 Preview lane is cooperative, with cancellation for the test render

The composer's preview worker receives requests over an mpsc channel and
drains the backlog to always work on the latest request. `render_frame`
takes an `Option<Arc<AtomicBool>>` "external running" flag — when the
caller flips it false, workers exit on their next iteration and the
function returns whatever's accumulated.

`Composer.preview_active` is shared with the preview worker. When the
render-time-estimate test fires (`maybe_run_render_test`), the App stores
`false` *before* spawning the test thread; the in-flight preview's workers
exit, the preview loop sees the cancel and skips writing partial garbage
into `preview_state` (so the previous good frame stays on screen). After
the test completes the App stores `true` and the next tick detects the
Running→Done transition and `mark_preview_dirty()`s so the preview
re-renders. The earlier "skip new previews while test is Running" gate in
`request_preview_if_dirty` stays as a second layer to prevent stale specs
from piling up in the channel.

### 4.7 Histograms are atomic; workers are oblivious to each other

`Vec<AtomicU32>` per pixel, `fetch_add` per scatter. `fetch_max` on
`max_count`. MH chains are per-worker (each has its own `chain_valid`,
`chain_contrib`, `chain_cr/ci`) so each core explores a different basin. No
merging overhead, no histogram-swap coordination.

Contention is bounded because the histogram has ~1M pixels; even hot pixels
get O(hundreds) of concurrent adds per second. Profiled fine on 32 cores.

---

## 5. Known limitations

Flagged here so the next iteration knows where the rough edges are.

- **`app.rs` is monolithic** (~3k lines). Splitting into `app/live.rs`,
  `app/composer.rs`, `app/dialog.rs`, etc., would reduce cognitive load. Not
  done because the rewrite churn isn't necessary for any specific feature.
- **PNG metadata isn't true EXIF**. `png` 0.17 has `add_text_chunk` /
  `add_itxt_chunk` but no ergonomic `eXIf` writer. Current scheme works with
  all mainstream inspection tools but a strict EXIF reader won't find the
  fields under `EXIF:`; they show up under `PNG:Textual` instead. Upgrading
  to `png` ≥0.18 or hand-writing the `eXIf` chunk via raw-chunk APIs would
  fix this; not done to minimize churn.
- **Onion skin is 2D-affine-only**. For rotations in planes like `XZ` or
  `YW` it foreshortens the image correctly; for `ZW` rotations it shows no
  motion (because ZW is "into the page" from the current view). Accurate —
  the user still gets gizmo feedback — but can read as "frozen" to someone
  unfamiliar. Fine for MVP.
- **Preview lane shares CPU with the UI sampler**. If the UI sampler is
  running at full throttle, preview renders compete. The `space` pause works
  around this manually; the auto-pause budget (§3.1) helps in practice
  because the live sampler stops on its own after ~10B proposals.
- **Video render + UI sampler also compete for CPU**. Same issue, different
  surface. Users can manually pause the UI with space.
- **Variable spacing is linear within each segment**. A long segment with
  high weight just spends more frames at the same Catmull-Rom-dictated
  speed; it doesn't "ease" into slow pans. Fine for motion control video
  but could be made fancier.
- **Timeline doesn't show time positions** (absolute seconds). It shows
  segments + weights. The dialog shows `duration = 2s × (n-1)`. Fine; just
  noting.
- **Live sampler workers don't reduce their count when paused** — they
  sleep in 50ms slices. Good for resume latency, wasteful for battery.
- **Auto-rotate keyframes** reuses the live view's `gamma` and the
  composer's `palette_override` (or live palette if none). Hardcoded 8
  frames per plane, 500k samples. Works but should expose knobs.
- **No provision for non-square aspect ratios on the gizmo**. Fixed square.
  Mostly fine.

---

## 6. Iteration backlog

Things that came up but are fine to defer:

- **Split app.rs**, per §5.
- **eXIf chunks** (real EXIF), per §5.
- ~~**Preview cancellability** — a cancel token through `worker_loop`.~~
  Done: `render_frame` accepts an external `running: Arc<AtomicBool>` and
  the composer uses it to suspend the preview during the render-time-
  estimate test (§4.6).
- **Nebulabrot** (3-channel histogram stratified by iteration band). The
  histogram abstraction already supports this; just needs a second dimension
  on the `AtomicU32` storage and a separate palette-apply path.
- **Save-view-to-JSON** export alongside the PNG. Currently a PNG is
  authoritative, but a human-readable export would make "send me a view"
  easier.
- **Bookmarks panel** in live mode for saved orientations/locations.
- **Multi-worker render with per-worker local histograms** merged
  periodically. Current atomic scheme is fine up to ~32 cores but higher
  counts may benefit.
- **In-UI error surface** for the composer — right now thumb load errors
  etc. go to stderr (now uniformly prefixed `error:` so they're greppable).
  Worth surfacing in the HUD too.
- **Mouse-wheel zoom in the preview pane**. Currently scroll only affects
  the live view.
- **Drop a keyframe from the composer's timeline via a right-click or a
  keystroke** instead of re-clicking its grid entry.
- **Render-dialog "apply" without close** so users can tweak and re-test.
- **Gap-drag affordance** — cursor doesn't change to indicate draggability.
- **CSV/JSON session export** for archival / sharing a composition.

---

## 7. Key files to read first for a new contributor

In order:

1. `Cargo.toml` — dep pin versions.
2. `main.rs` + `cli.rs` — entry points, argv dispatch.
3. `view.rs` — the 4D viewing math. Everything else references this.
4. `sampler.rs` + `orbit.rs` + `render.rs` — how a frame gets computed.
5. `videorender.rs` + `session.rs` — how a video gets rendered, including
   the interpolation helpers and the resume contract.
6. `app.rs` — the UI. `run()` → `App::run_app` → `App::window_event` →
   `App::redraw`. Look for the `Mode::{Live,Composer}` branches.
7. `composer.rs` — mostly just state + the preview worker loop.

---

## 8. Recent session intent (handoff note)

The last working session implemented (in this order):

- Tabbed UI, composer skeleton, thumbnail browser, timeline with arrow bars.
- Composer controls: palette override, auto-rotate, render timeline.
- Unified render dialog (composer feeds the same dialog as live; both take
  their keyframes from the dialog state).
- Persistent top-right render-status widget + render queue.
- Draggable timeline items (reorder).
- Interpolated preview render + auto-advance playback.
- Variable spacing (drag gaps → weights → preview + video mapping).
- **Timeline state save/load** (`timeline_save.rs`, §3.2): autosave slot +
  manual timestamped snapshots + a top-left save/load button strip. Missing
  keyframes are surfaced in-timeline (dim + red X) and block render spawns
  with a named-file error list. Triggered by a render failure caused by
  `current_exe()` returning `/path/to/bin (deleted)` after an in-session
  rebuild — the UI had a ready-to-render timeline and no way to persist it.

All of this is working-state but not deeply polished. Next session will
iterate on design in conversation before making further changes.

### 8.1 Subsequent session

- **Auto-pause** UI sampler at 10B samples; `invalidate()` re-arms and
  unpauses on every camera move.
- **Right-drag** added as a separate "camera-orbit" rotation mode
  (`(XZ, YZ)` plain, `(XW, YW)` with shift); both axes inverted.
- Help overlay anchored bottom-left instead of centered.
- **Composer palette editor** finished: reset button + click-marker-to-edit
  (the no-move-promote-to-Click path was previously dead code).
- **Composer render panel**: scale slider replaced by a 6-preset resolution
  radio grid (4K/1440p/1080p/720p/480p/240p). Tab cycles between W/H
  fields. Active-render row shows live ETA derived from observed frame
  timing; the static `estimate:` line above is reserved for the test
  measurement (next-render predictor). Per-keyframe tabs in the dialog also
  got the same Tab-cycles-fields fix.
- **Composer preview gamma** slider added under the palette area, range
  `[0.1, 2.5]`, applies between histogram and palette via `palette::apply`.
  Overrides per-keyframe gamma during preview only — final video render
  still interpolates per-keyframe gamma.
- **Render-time-estimate test cancels in-flight preview** (§4.6) so the
  measurement is repeatable across clicks.
- **`half_width` interpolation** changed from log-cr to linear-cr so all
  parameters share the same progression curve through a segment (§3.4).
  **Reverted**: linear Hermite overshoots below zero whenever a zoomed-out
  keyframe sits next to a much deeper one (e.g. KF0 half_width 2.09 →
  KF1 0.077). The `1e-12` clamp kept the renderer from dividing by zero
  but collapsed pixel_pitch, producing a blank frame near the start of
  the affected segment. Back to `log(half_width)` interpolation.
- **Use in Live** added as the first item in the grid context menu;
  "delete (move to trash)" renamed to "Trash".
- **Render-pipeline failure logging**: every `RenderResult::Failed` writer
  pairs with an `eprintln!("error: …")` so failures don't silently vanish
  into UI state. `spawn_render_job_free`'s three `?` steps each log which
  one tripped.

### 8.2 Subsequent session

- **Live-mode palette / gamma HUD** (lower-left) added — same widgets as
  the composer's preview pane, stacked vertically: gradient bar with
  editable stops + reset button, then a gamma slider underneath. Live
  rendering now flows through an editable `Palette` rebuilt from
  `palette_stops` instead of indexing a fixed `Vec<Palette>`. The `p` /
  shift-`p` cycle keys load preset stops via the new
  `palette::builtin_stops()` helper (refactored from `palette::builtin()`
  so the same data feeds both the LUT-only `Vec<Palette>` and the editable
  stop list).
- **Help overlay** moved from bottom-left to top-center (just below the
  tab bar) so it doesn't cover the new palette HUD when toggled.
- **Live HUD interaction**: `LiveWidgetDrag` enum (`None | Stop(idx) |
  Gamma | ColorR/G/B`) plus `live_widget_press_origin` track active
  drags so the press handler can intercept clicks inside the panel
  before the rotation/pan path. Release promotes a no-move stop click to
  "open color popover" (mirrors the composer's `did_move` check). Color
  popover anchors above the panel so it can't push offscreen.
