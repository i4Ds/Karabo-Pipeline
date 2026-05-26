# SKA-SDP Functionalities To Test

## Where to find code examples
- Simulation backend usage (`run_simulation(..., backend=...)`):
  - `karabo/examples/Sky_Simulation.ipynb`
  - `karabo/examples/Sky_Simulation_SDP.ipynb`
- Imaging backend switch (`ImagingBackend`, `get_imager`, `invert`, `restore`):
  - `karabo/examples/imaging.ipynb`
  - `karabo/examples/source_detection.ipynb`
  - `karabo/examples/source_detection_big_files.ipynb`
- Source detection on SDP-produced images:
  - `karabo/examples/source_detection.ipynb`
  - `karabo/examples/source_detection_big_files.ipynb`
- Docs pages for backend selection:
  - `doc/src/main_features/imaging_backend_selection.rst`
  - `doc/src/main_features/simulation_backend_selection.rst`

## Core simulation
- Run simulation with `SimulatorBackend.SDP` through `InterferometerSimulation.run_simulation(...)`.
- Verify simulation with and without primary-beam input.
- Verify telescope + sky + observation setup works in normal notebooks/scripts.

## Core imaging (with backend-switched path)
- Select imaging backend via `ImagingBackend.SDP`.
- Run `get_imager(...).invert(...)` and confirm dirty + PSF outputs are produced.
- Run `restore(...)` and confirm restored output is produced.

## Imaging output
- Check image dimensions consistency (dirty, PSF, restored).
- Check PSF looks physically plausible.
- Check dirty/restored images are visually plausible.

## Source-detection
- Run source-detection workflows using SDP-generated dirty images.
- Confirm detection/evaluation steps complete without backend-related failures.
- Confirm detection outputs are scientifically plausible.

## What to report
- If the functionality is working correctly
- Are the output plausible ?
- Error traceback (if any) / Output problems (plot/image path or screenshot)
