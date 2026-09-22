# Third-Party Notices & Licensing

## Spine Runtimes & Spine Data Format

This project outputs skeleton data in the **Spine 3.8.99 format** (`skeleton.json` + `skeleton.atlas`), designed to be imported directly into the Spine Editor or loaded via official Spine Runtimes.

Spine and Spine Runtimes are products of **Esoteric Software LLC**.
For licensing terms regarding Spine and the Spine Runtimes, see the [Spine Runtimes License Agreement](https://esotericsoftware.com/spine-runtimes-license).

### Key Terms & Obligations

1. **unity2spine's own code is open source under the MIT License** (see [LICENSE](LICENSE)).
2. **Output format**: unity2spine emits standard Spine 3.8 JSON skeleton and texture atlas files.
3. **Spine Editor Import**: Importing the generated skeleton files into the official Spine Editor requires a valid **Spine Editor license** purchased from Esoteric Software LLC.
4. **Product Integration**: Using official Spine Runtimes to render Spine skeleton data in games or interactive commercial applications requires each developer/product user to comply with Esoteric Software's licensing terms.

> **Notice**: unity2spine is an independent open-source converter tool. It is neither affiliated with, endorsed by, nor sponsored by Esoteric Software LLC. Using or redistributing Spine skeleton data or runtimes in commercial applications requires adherence to Esoteric Software's official licensing policies.

## Third-Party Libraries

This tool utilizes the following open-source Python libraries:

- **UnityPy**: MIT License (Unity asset bundle unpacking and parsing)
- **PyQt5**: GPL v3 / Riverbank Commercial License (Desktop GUI frontend)
- **PyOpenGL**: BSD License (Hardware-accelerated viewport rendering)
- **Flask**: BSD-3-Clause License (Local Web preview server)
- **Pillow**: HPND License (Image processing)
- **OpenCV (opencv-python)**: Apache 2.0 License (Image manipulation and rasterization)
- **NumPy**: BSD-3-Clause License (Matrix math, transformations, and curve evaluation)
