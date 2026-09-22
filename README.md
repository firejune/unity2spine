# unity2spine

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="license" /></a>
  <img src="https://img.shields.io/badge/python-3.10+-brightgreen.svg?style=flat-square" alt="python version" />
  <img src="https://img.shields.io/badge/spine%20format-3.8.99-orange.svg?style=flat-square" alt="spine format" />
  <img src="https://img.shields.io/badge/gui-PyQt5%20%2B%20OpenGL-purple.svg?style=flat-square" alt="GUI" />
</p>

**Convert Unity 2D rigged AssetBundles into Spine 3.8 skeleton data.** Extracts bones, weighted meshes, sprites, and animation clips directly from Unity bundles and compiles them into clean `skeleton.json` + `skeleton.atlas` files that load in any Spine 3.8 runtime and import cleanly into the Spine Editor.

Includes a desktop GUI converter with real-time OpenGL animation preview, a local web-based preview server, an automated GIF rasterizer, and a mathematical pose oracle to verify converted accuracy against Unity ground truth.

---

## What You Get

| You have | You run | You get |
| --- | --- | --- |
| A Unity 2D AssetBundle (`__data` or bundle file) | `unity2spine path/to/__data` | `spine_editor/` with `skeleton.json`, `skeleton.atlas`, and cut PNG images ready for Spine Editor import |
| An AssetBundle and need runtime texture pages | `unity2spine path/to/__data --runtime` | `spine/` with atlas-packed texture pages for Spine Runtimes |
| A desire to preview or batch-convert without coding | `unity2spine-gui` | A dark-mode desktop GUI: drag & drop files/folders, batch export, timeline scrubbing, and bone display |
| An AssetBundle you want to preview in browser | `unity2spine-viewer path/to/__data` | A local Flask server serving a self-contained Spine Web viewer at `http://localhost:5000` |
| Animations you want to export as lightweight GIFs | `unity2spine path/to/__data --gif` | Rasterised `.gif` files for every `AnimationClip` in the bundle |
| Ground-truth verification needs | `python oracle.py <bundle> <skeleton.json>` | Exact degree-by-degree, bone-by-bone rotational and positional comparison against Unity |

---

## Features

- **End-to-End AssetBundle Extraction**: Directly unpacks Unity assets using `UnityPy` without requiring the Unity Editor or custom C# export scripts.
- **SkinnedMeshRenderer & SpriteRenderer Support**:
  - Full vertex weight skinning (up to 4 bone influences per vertex).
  - Preserves bind poses, UV coordinates, and triangle topology.
  - Correct draw ordering and slot creation based on Unity sorting layers and hierarchy depth.
- **High-Fidelity Animation Curve Sampling & Bezier Fitting**:
  - Decodes `AnimationClip` curves: Transform Euler rotations (`kBindTransformEuler`), positions, scales, renderer color/alpha tinting, and GameObject active/visibility states.
  - Automatically fits curves into Spine 3.8 cubic Bezier timeline segments.
  - Generalized support for facial cross-fades and blend-shape opacity curves.
- **Desktop GUI Converter (`gui_converter.py` / `unity2spine-gui`)**:
  - Drag & drop single files, multiple files, or entire directories.
  - Smart folder scanner: recursively discovers valid character/rig bundles while skipping non-bundle files.
  - Real-time animation player powered by hardware-accelerated OpenGL skinning.
  - Play, pause, scrub, loop, speed control (0.1x - 5.0x), and bone overlay toggle.
- **Web-Based Spine Viewer (`viewer_server.py` / `unity2spine-viewer`)**:
  - Built-in Flask server and lightweight WebGL/Canvas renderer.
  - Interactive slot/part visibility toggles and animation switcher.
- **GIF Rasterizer**:
  - Multi-threaded rendering of skeleton animations straight to transparent or solid-background GIFs.
- **Pose Oracle & Quantitative Fidelity Verification**:
  - Evaluates original Unity 4x4 transform matrices against Spine's 2D world transform mathematics.
  - Measures rotation error in degrees and position drift as percentage of rig dimensions.

---

## Installation

### Prerequisites

- Python 3.10 or later
- macOS, Linux, or Windows

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/firejune/unity2spine.git
cd unity2spine

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

Alternatively, install in editable mode to get CLI commands in your PATH:

```bash
pip install -e .
```

---

## Quick Start

### 1. Command Line Interface (CLI)

```bash
# Basic conversion (creates <parent>/spine_editor/ for Spine Editor import)
unity2spine path/to/__data

# Specify a custom output directory
unity2spine path/to/__data --output ./output/my_character

# Export with runtime atlas layout (shared packed texture pages)
unity2spine path/to/__data --runtime

# Export Spine skeleton AND render GIF animations
unity2spine path/to/__data --gif

# Export ONLY GIF animations at 720px width
unity2spine path/to/__data --gif-only --gif-width 720
```

### 2. Desktop GUI Converter

Launch the PyQt5 GUI:

```bash
unity2spine-gui
# Or: python gui_converter.py
```

- **Drag & drop** any Unity AssetBundle file into the window to preview its animations instantly.
- Click **"📁 Smart Scan Folder..."** to scan an entire game directory: `unity2spine` automatically detects bundles containing character rigs (`SkinnedMeshRenderer` / `SpriteRenderer`) and queues them for batch conversion.
- Use the preview timeline on the right to scrub frames, change playback speed, or toggle bone display.

### 3. Web-Based Preview Server

```bash
unity2spine-viewer path/to/__data --port 5000
# Or: python viewer_server.py path/to/__data
```

Open `http://localhost:5000` in your browser to inspect the skeleton, switch animations, and toggle individual slot visibilities.

---

## CLI Options & Reference

```
usage: unity_to_spine.py [-h] [--output OUTPUT] [--runtime] [--fps FPS]
                         [--target-w TARGET_W] [--gif] [--gif-only]
                         [--gif-width GIF_WIDTH]
                         bundle

positional arguments:
  bundle                Path to Unity AssetBundle file or directory (typically '__data')

options:
  -h, --help            Show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Custom output directory (default: <bundle-parent>/spine_editor/ or spine/)
  --runtime             Export in runtime layout (atlas-packed texture pages) instead of Editor import layout
  --fps FPS             Animation sampling frame rate (default: 30)
  --target-w TARGET_W   Target canvas width for Spine coordinate space (default: 1600)
  --gif                 Rasterise animations to GIF alongside Spine export
  --gif-only            Only rasterise GIFs without writing Spine skeleton files
  --gif-width GIF_WIDTH
                        Width in pixels for generated GIF frames (default: 720)
```

### Output Directory Structure

#### Editor Import Layout (Default: `spine_editor/`)
Designed for direct import into the Spine Editor (**Spine Menu → Import Data**):
```
spine_editor/
├── skeleton.json      # Spine 3.8 skeleton data
├── skeleton.atlas     # Region mapping for loose parts
└── images/            # Individual PNG parts
    ├── Head.png
    ├── Body.png
    └── ...
```

#### Runtime Layout (`--runtime`: `spine/`)
Designed for direct loading in game engines using Spine Runtimes:
```
spine/
├── skeleton.json      # Spine 3.8 skeleton data
├── skeleton.atlas     # Atlas definition
└── skeleton.png       # Packed texture atlas page(s)
```

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `SPINE_FPS` | `30` | Animation curve sampling frequency in frames per second. |
| `SPINE_TARGET_W` | `1600` | Target viewport width used to calibrate world scale. |
| `GIF_W` | `720` | Default pixel width for exported GIF animations. |
| `GIF_FPS` | `24` | Frame rate for exported GIF animations. |
| `GIF_BG` | `ffffff` | Hex background color for GIF output (`ffffff` = white). |
| `GIF_WORKERS` | `0` | Thread count for GIF rasterization (`0` = auto-detect CPU cores, max 8). |

---

## Architecture & Conversion Pipeline

```
[Unity AssetBundle]
       │
       ▼
 1. UnityPy Asset Decoder
    ├── Read GameObject & Transform hierarchies
    ├── Extract SkinnedMeshRenderer (vertices, UVs, 4-bone weights, bindpose)
    ├── Extract SpriteRenderer (sprites, sort orders)
    └── Read AnimationClip (Euler angles, translation, scale, alpha, active curves)
       │
       ▼
 2. Coordinate & Transform Mapping
    ├── Left-handed 3D Unity coords  ──►  Right-handed 2D Spine coords
    ├── Decompose 4x4 matrices (x, y, rotation, scaleX, scaleY, shearX, shearY)
    └── Bake uniform root scale calibrated to TARGET_W
       │
       ▼
 3. Timeline Fitting
    ├── Sample dense Unity animation curves at SPINE_FPS
    ├── Detect stepped transitions (GameObject active, component enabled)
    └── Compute cubic Bezier control points for rotation, translate, scale & shear
       │
       ▼
 4. Spine 3.8 Emitter
    ├── skeleton.json (bones, slots, skins, attachments, animations)
    ├── skeleton.atlas (texture regions and page coordinates)
    └── Texture rasterization / image slicing
```

### Key Technical Details

- **Coordinate Normalization**: Unity evaluates transforms in 3D left-handed space ($+X$ right, $+Y$ up, $+Z$ forward), while Spine evaluates in 2D right-handed planar space. Rotations and shears are decomposed using single-convention affine decomposition (`decompose2d`), ensuring angular parity across parent-child chains.
- **Hierarchy Reconstruction**: Unity rigs frequently attach mesh renderers to leaf nodes separate from the bone deformer tree. `unity2spine` reconstructs a clean 2D skeletal hierarchy where bone transforms and slot attachments are correctly decoupled.
- **Facial Blend Shapes & Cross-Fades**: Models with dual mouth setups (e.g. idle vs. speaking) or blush layers driven by blend shape weights (`customType 22`) are automatically resolved to opacity and slot color multiplier curves.

---

## Validation: Pose Oracle & Fidelity

To verify that the emitted Spine data faithfully recreates the original animation, this repository includes an independent mathematical oracle (`oracle.py`):

1. **Unity Side (Ground Truth)**: Samples `AnimationClip` curves directly from the bundle, applies them to the Unity `Transform` tree, and computes the ground-truth world 4x4 matrix for every bone at every keyframe.
2. **Spine Side (Evaluator)**: Runs a standalone evaluator implementing Spine 3.8 world transform mechanics (`Bone.updateWorldTransform`) with Bezier timeline interpolation.
3. **Comparison**: Both evaluate to world space and decompose through the exact same matrix math.

```bash
# Run pose oracle on a single converted character
python oracle.py path/to/__data path/to/spine_editor/skeleton.json --samples 8

# Run oracle across multiple rigs and write fidelity metrics
python oracle_all.py <bundles-dir> <output-dir> <output-dir> results.json
python fidelity.py "results.json" probe_3d.json fidelity.json
python fidelity_md.py fidelity.json fidelity.md
```

Across production character corpora, `unity2spine` achieves **sub-degree angular precision** ($< 0.1^\circ$ median bone error) across non-planar rigs.

---

## Licensing & Third-Party Notices

- **unity2spine** code is licensed under the [MIT License](LICENSE).
- **Spine Skeleton Data & Runtimes**: This tool outputs files formatted for Spine 3.8 (`skeleton.json` / `skeleton.atlas`). Spine and Spine Runtimes are copyrighted products of **Esoteric Software LLC**. Using official Spine Runtimes or importing generated data into the Spine Editor requires an appropriate license from Esoteric Software LLC. See [NOTICE.md](NOTICE.md) for full details.
