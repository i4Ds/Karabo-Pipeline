# Installation (macOS, Apple Silicon)

Karabo can run natively on Apple Silicon Macs (M1 to M5). This does not use
Rosetta or an amd64 container. There is no conda package for osx-arm64 yet,
so the install script builds the missing pieces.

## Requirements
- macOS on Apple Silicon, in a native arm64 shell (not under Rosetta)
- [Miniforge](https://github.com/conda-forge/miniforge) (`mamba` or `conda`)
- [Homebrew](https://brew.sh), Xcode command line tools, `git`
- About 6 GB disk space

## Install

```shell
git clone https://github.com/i4Ds/Karabo-Pipeline.git
cd Karabo-Pipeline
bash macos/install_osx_arm64.sh
conda activate karabo-osx
```

The script does the following:

1. It creates the conda env `karabo-osx` from `environment-osx-arm64.yaml`.
   Set `KARABO_ENV` to use another name.
2. It installs WSClean 3.7 from the `mwatelescope/tap` Homebrew tap
   (arm64 bottle), unless `wsclean` is already on `PATH`.
3. It builds OSKAR 2.12.2 from source into the env with OpenCL, so that it can
   use the Mac GPU. It also builds OSKAR's Python bindings. It applies the
   same one-line fix to OSKAR as the i4ds conda build (`macos/patches/`).
4. It builds `ska-sdp-func` 1.0.0 from source, CPU only.
5. It installs `bdsf`, `MontagePy`, `tools21cm`, `ska-sdp-datamodels` and
   `ska-sdp-func-python` with pip `--no-deps`. Their dependencies come from
   conda.
6. It installs Karabo in editable mode.

The build tree is `.build-osx-arm64/` in the repository. It is git-ignored.

## GPU (Apple Silicon)

OSKAR runs interferometer simulations on the Mac GPU through OpenCL. Apple
GPUs support single precision only. With the default `use_gpus=None`, Karabo
asks OSKAR whether an OpenCL GPU exists:

- `precision="single"` (the default): the simulation runs on the GPU.
- `precision="double"`: the simulation runs on the CPU.

If you set `use_gpus=True` with double precision, OSKAR prints
"No GPU capability available" and also uses the CPU.

On an M5 Pro, MeerKAT with 47,900 GLEAM sources, 64 channels and 48 time
steps took 52 s on the GPU and 290 s on the CPU. The visibilities agree to
2.5e-5 relative, which is the single-precision level.

## What does not work on macOS
- CUDA does not exist on macOS. `bluebild`, `ska-gridder-nifty-cuda` and the
  SDP `wg` imaging context (cupy) are not available.
- WSClean and the SDP imagers run on the CPU.
- WSClean is built without EveryBeam and IDG.
