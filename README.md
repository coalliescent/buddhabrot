# buddhabrot

Interactive 4D buddhabrot explorer with a keyframe-based composer and video
render pipeline. Written in Rust, CPU-only, no external runtime besides
`ffmpeg` for video assembly.

The buddhabrot is the orbit-density visualization of the Mandelbrot set. This
project lifts it from the classic 2D c-plane into the full 4D state space
`(c_re, c_im, z_re, z_im)`, with a rotatable SO(4) orientation picking the
viewing plane. Drag to rotate through any of six 2D rotation planes, scroll
to zoom, save frames as PNGs, then drop them on a timeline and render an
interpolated video.

See [DESIGN.md](DESIGN.md) for architecture, design intents, and module map.

## Features

### Live mode
- Real-time Metropolis-Hastings sampler across all CPU cores, atomic
  histogram, palette-applied each frame.
- Six rotation planes (XY / XZ / XW / YZ / YW / ZW) mapped to mouse +
  modifier; right-drag is a camera-style orbit. Pan with middle drag, zoom
  with scroll (mouse-anchored).
- Tesseract orientation gizmo (24 face snaps) with eased rotation animation.
- Onion-skin overlay during drag for spatial continuity while the live
  histogram catches up.
- Palette / gamma HUD with editable stops (click to add, drag to move,
  click for an RGB popover, right-click to delete) and preset cycling.
- Undo / redo of view state (256-deep, time-coalesced).
- Auto-pause after ~10B sample proposals so a finished view doesn't keep
  burning CPU; any camera move re-arms.
- Save current view as a PNG with `Buddhabrot.*` tEXt metadata that fully
  reconstructs the `FrameSpec` for re-render.

### Composer mode
- Thumbnail grid of every saved PNG in `~/buddhabrot/`, decoded and
  downsampled in the background.
- Timeline: click thumbs to add/remove, drag to reorder, drag the gaps
  between items to weight segment duration. Per-gap delta arrows show
  rotation / pan / zoom magnitude at a glance.
- Live interpolated preview pane with playhead scrubbing, play/pause,
  and the same palette + gamma editor as the live HUD (overrides
  per-keyframe values during preview).
- Auto-generate keyframes: ring of frames around each of the 6 SO(4)
  rotation planes from the current live view.
- Embedded render panel with W/H fields, samples/frame slider, resolution
  preset radios (4K / 1440p / 1080p / 720p / 480p / 240p), test-render ETA
  prediction, and a queue showing active progress + per-frame timing.
- Timeline persistence: autosave to `~/buddhabrot/timeline.state` plus
  named manual snapshots in `~/buddhabrot/timelines/`. Missing keyframe
  PNGs are surfaced in-timeline and block render spawns.

### Render pipeline
- Detached subprocess per render — UI can crash or be closed without losing
  in-flight work; sessions are resumable via `--resume <session_dir>`.
- On-disk session schema (`session.json` + `frames/NNNNN.png`) shared
  between CLI and UI; the UI scans `~/buddhabrot/.render-*/` on startup
  and adopts live sessions or offers to resume dead ones.
- Catmull-Rom interpolation across SO(4) matrices (with Gram-Schmidt
  reorthonormalization), pan, `log(half_width)`, gamma, max-iter; LUTs
  are linearly blended.
- Variable per-segment durations and per-keyframe Hermite tangent tensions.
- ffmpeg subprocess assembles `frames/%05d.png` → `libx264` MP4.

### CLI
```
buddhabrot                              open the interactive window
buddhabrot render-frame --in KF.png --out out.png [--samples N --width W --height H --threads K]
buddhabrot render-video --out vid.mp4 KF1.png KF2.png ...
                       [--fps N --duration SEC --frames N --width W --height H
                        --samples N --threads K --segments CSV --tensions CSV
                        --gamma F --palette-stops CSV --rho F --resume DIR]
```
Same code path as the UI's render queue.

## Build & run

```
cargo run --release
```

Requires a system `ffmpeg` on `PATH` for video render.

## Room for improvement

Directions worth iterating on next:

- **Render performance tuning** — the atomic-histogram MH sampler is
  fine to ~32 cores but contention and cache traffic likely dominate
  beyond that; per-worker local histograms with periodic merges, batch
  sizing, and the warmup/burn-in ratio are all worth profiling.
- **UI rationalization** — replace context menus and ad-hoc click
  patterns with first-class affordances; surface render errors in the
  HUD instead of stderr; cursor changes for draggable gaps; keystroke
  removal of timeline items.
- **Render timeline editability / improvements** — show absolute time
  positions, per-segment easing curves (not just linear within a
  Catmull-Rom segment), drag-to-trim, multi-select.
- **Render spline path improvements** — better SO(4) interpolation
  (true geodesic / log-quaternion-style) instead of matrix-entry CR
  with reorthonormalization; smarter zoom pacing that doesn't need
  the `log(half_width)` workaround for adjacent shallow/deep keyframes;
  exposable per-segment tension and bias.
- **Live view coordinate-space navigability** — drag-modifier mapping
  to rotation planes is unintuitive without the help overlay; some
  visual indication of which plane(s) the current drag is rotating
  through, and a less mode-heavy way to pick a plane.
- **Better render palette management** — saved/named palettes,
  import/export, per-segment palette transitions independent of
  per-keyframe metadata, A/B comparison.
- **Multiplatform app** — currently developed on Linux; Windows and
  macOS need testing for paths (`~/buddhabrot`), file dialogs, ffmpeg
  discovery, and high-DPI handling. A packaged build would help.
