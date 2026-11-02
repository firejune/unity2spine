# Contributing

`unity2spine` converts 2D skeletal animations inside Unity AssetBundles (`MeshRenderer`, `SkinnedMeshRenderer`, `SpriteRenderer`, and `AnimationClip` hierarchies) into Spine 3.8.99 format, bundled with a PyQt5 desktop converter and a browser-based Spine WebGL player. Read [README.md](README.md) and [NOTICE.md](NOTICE.md) before submitting changes.

## Issues first

Open an issue before starting substantial changes to transform math, curve fitting, mesh baking, or viewer architecture. Good defect reports include:

- Python version (`python --version`), OS, and `UnityPy` / `numpy` / `scipy` versions
- the exact CLI command or GUI option used
- output from the pose oracle (`python oracle.py <bundle> <skeleton.json> --samples 8`) if the issue involves bone transforms, rotation flipping, or mesh deformation

⚠️ **Never upload, attach, or commit proprietary or copyrighted game AssetBundles, extracted character textures, or commercial Spine assets to issues or pull requests.** Use minimal synthetic test cases or redistributable fixtures when reproducing bugs.

## Before you open a pull request

`unity2spine` requires **Python 3.10+**. Install dependencies and verify your changes from the repository root:

```bash
pip install -r requirements.txt
python unity_to_spine.py --help
```

When modifying coordinate conversion, affine decomposition, skinning, or animation curve resampling in `unity_to_spine.py`, always run the mathematical pose oracle against a test bundle and confirm zero regressions in bone world positions and angles:

```bash
python oracle.py <bundle_path> <output_dir>/<skeleton>.json --samples 8
```

## What a change has to clear

- **Mathematical transform fidelity**: preserve the single-convention 2D affine decomposition (`decompose2d`) bridging Unity's 3D left-handed coordinate space and Spine 3.8.99's 2D right-handed planar hierarchy. Transform or curve changes must not degrade `oracle.py` pose accuracy.
- **Headless CLI independence**: `unity_to_spine.py`, `oracle.py`, and `viewer_server.py` must run without requiring desktop GUI libraries (`PyQt5` and `PyOpenGL` are optional dependencies for `gui_converter.py`).
- **Clean repository tree**: never commit generated `output*` directories, extracted PNG atlases, `.bundle` files, or third-party proprietary assets.
- **License boundaries**: respect the Spine Runtimes licensing terms documented in [NOTICE.md](NOTICE.md).

## Commits and pull requests

Use concise, descriptive commit subjects (`feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`). Keep each pull request focused on a single concern and reference the tracking issue (`Closes #N`).

## License

Contributions to the original code in this repository are accepted under the MIT License in [LICENSE](LICENSE). See [NOTICE.md](NOTICE.md) for third-party notice details.
