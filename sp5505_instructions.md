your task is to create a docker image `sp5505.Dockerfile` that:
1. is based on, or has the same functionality as `quay.io/jupyter/minimal-notebook:notebook-7.2.2` (ipykernel etc.)
2. has a python environment that can be seen by jupyterhub, and is activated by default in the shell of `USER ${NB_UID}`
3. has `karabo-pipeline` installed in the python environment
4. passes all karabo-pipeline tests `python -m pytest .`
5. python 3.10 would be nice, 3.9 is fine too
6. install most dependencies via spack instead of pip or conda

# some info about the current conda environment that should be replaced by spack, for details, see conda_list.txt
# astropy                   5.1.1           py310hde88566_3    conda-forge
# astropy-base              5.1.1                hd2eee37_0    conda-forge
# astropy-healpix           1.1.2           py310hf462985_0    conda-forge
# bdsf                      1.10.2          py310h66aa11f_1002    i4ds
# bluebild                  0.1.0           py310h3fd9d12_4    i4ds
# dask                      2022.12.1          pyhd8ed1ab_0    conda-forge
# dask-core                 2022.12.1          pyhd8ed1ab_0    conda-forge
# dask-memusage             1.1                        py_0    conda-forge
# dask-mpi                  2022.4.0           pyh458ca06_3    conda-forge
# h5py                      3.13.0          mpi_mpich_py310hec5fa98_0    conda-forge
# harp                      1.22            py310hf6a41b2_0    conda-forge
# hdf4                      4.2.15               h9772cbc_5    conda-forge
# hdf5                      1.14.3          mpi_mpich_h0f54ddc_5    conda-forge
# healpy                    1.16.6          py310h510b526_2    conda-forge
# libblas                   3.9.0           34_h59b9bed_openblas    conda-forge
# libboost                  1.82.0               h6fcfa73_6    conda-forge
# libboost-devel            1.82.0               h00ab1b0_6    conda-forge
# libboost-headers          1.82.0               ha770c72_6    conda-forge
# libboost-python           1.82.0          py310hcb52e73_6    conda-forge
# libboost-python-devel     1.82.0          py310h17c5347_6    conda-forge
# libbrotlicommon           1.1.0                hb03c661_4    conda-forge
# libbrotlidec              1.1.0                hb03c661_4    conda-forge
# libbrotlienc              1.1.0                hb03c661_4    conda-forge
# libcblas                  3.9.0           34_he106b2a_openblas    conda-forge
# libclang13                20.1.8          default_ha444ac7_0    conda-forge
# libcufft                  10.7.2.50            h80a1efe_0    nvidia/label/cuda-11.7.0
# libcurl                   8.15.0               hc1efc7f_0
# montagepy                 6.0.0           py310hf570258_0    i4ds
# mpi                       1.0.1                     mpich    conda-forge
# mpi4py                    4.1.0           py310hab64184_102    conda-forge
# mpich                     4.3.1              h74c0cd0_102    conda-forge
# pytest                    8.4.1              pyhd8ed1ab_0    conda-forge
# pytest-arraydiff          0.6.1              pyhd8ed1ab_1    conda-forge
# pytest-astropy            0.11.0             pyhd8ed1ab_1    conda-forge
# pytest-astropy-header     0.2.2              pyhd8ed1ab_1    conda-forge
# pytest-cov                6.2.1              pyhd8ed1ab_0    conda-forge
# pytest-doctestplus        1.4.0              pyhd8ed1ab_0    conda-forge
# pytest-filter-subpackage  0.2.0              pyhd8ed1ab_1    conda-forge
# pytest-mock               3.14.1             pyhd8ed1ab_0    conda-forge
# pytest-remotedata         0.4.1              pyhd8ed1ab_0    conda-forge
# pytest-runner             6.0.0              pyhd8ed1ab_0    conda-forge
# python                    3.10.18              h1a3bd86_0
# python-casacore           3.5.2           py310hc6ae7ae_3    conda-forge
# python-dateutil           2.9.0.post0        pyhe01879c_2    conda-forge
# python-fastjsonschema     2.21.2             pyhe01879c_0    conda-forge
# python-flatbuffers        25.2.10            pyhbc23db3_0    conda-forge
# python-json-logger        2.0.7              pyhd8ed1ab_0    conda-forge
# python_abi                3.10                    2_cp310    conda-forge
# pytz                      2025.2             pyhd8ed1ab_0    conda-forge
# pyuvdata                  2.4.2           py310hcc13569_0    conda-forge
# pywavelets                1.8.0           py310hf462985_0    conda-forge
# scikit-image              0.25.0          py310h5eaa309_0    conda-forge
# scikit-learn              1.7.1           py310h228f341_0    conda-forge
# scipy                     1.13.1          py310h93e2701_0    conda-forge
# ska-gridder-nifty-cuda    0.3.0           py310he9e06a5_1    i4ds
# ska-sdp-datamodels        0.1.3           py310h410524c_1    i4ds
# ska-sdp-func              0.0.6           py310he53d0f1_0    i4ds
# ska-sdp-func-python       0.1.4           py310h410524c_2    i4ds

# conda tree leaves
# montagepy
# bluebild
# rascil
# libcufft
# cudatoolkit
# cuda-version
# oskarpy
# cuda-cudart

# current main issue:

```bash
docker build -f sp5505.Dockerfile --progress=plain . 2>&1 | tee sp5505.log

# RUN if [ "0" = "1" ]; then exit 0; fi;     export OMP_NUM_THREADS=1 &&     export OPENBLAS_NUM_THREADS=1 &&     export MKL_NUM_THREADS=1 &&     export NUMEXPR_NUM_THREADS=1 &&     pytest -q -x --tb=short -k "not test_suppress_rascil_warning and not (oskar or OSKAR)" /home/jovyan/Karabo-Pipeline &&     rm -rf /home/jovyan/.astropy/cache            /home/jovyan/.cache/astropy            /home/jovyan/.cache/pyuvdata            /home/jovyan/.cache/rascil
# ..F
# =================================== FAILURES ===================================
# _______________ test_gaussian_beam[SimulatorBackend.RASCIL-MID] ________________
# Karabo-Pipeline/karabo/test/conftest.py:26: in _dtdb_safe
#     return _orig_dtdb(date1, date2, ut, elong, u, v)
# /opt/view/lib/python3.10/site-packages/erfa/core.py:16572: in dtdb
#     c_retval = ufunc.dtdb(date1, date2, ut, elong, u, v)
# E   ValueError: Invalid data-type for array
#
# During handling of the above exception, another exception occurred:
# Karabo-Pipeline/karabo/test/test_000_astropy_env.py:22: in _dtdb_safe
#     return _orig_dtdb(date1, date2, ut, elong, u, v)
# Karabo-Pipeline/karabo/test/conftest.py:36: in _dtdb_safe
#     return _orig_dtdb(date1, date2, ut, elong, u, v)
# /opt/view/lib/python3.10/site-packages/erfa/core.py:16572: in dtdb
#     c_retval = ufunc.dtdb(date1, date2, ut, elong, u, v)
# E   ValueError: Invalid data-type for array
#
# During handling of the above exception, another exception occurred:
# Karabo-Pipeline/karabo/test/conftest.py:26: in _dtdb_safe
#     return _orig_dtdb(date1, date2, ut, elong, u, v)
# /opt/view/lib/python3.10/site-packages/erfa/core.py:16572: in dtdb
#     c_retval = ufunc.dtdb(date1, date2, ut, elong, u, v)
# E   ValueError: Invalid data-type for array
#
# During handling of the above exception, another exception occurred:
# Karabo-Pipeline/karabo/test/test_beam.py:150: in test_gaussian_beam
#     visibility = simulation.run_simulation(
# Karabo-Pipeline/karabo/simulation/interferometer.py:514: in run_simulation
#     return self.__run_simulation_rascil(
# Karabo-Pipeline/karabo/simulation/interferometer.py:585: in __run_simulation_rascil
#     vis = create_visibility(
# /opt/view/lib/python3.10/site-packages/ska_sdp_datamodels/visibility/vis_create.py:148: in create_visibility
#     stime = calculate_transit_time(
# /opt/view/lib/python3.10/site-packages/ska_sdp_datamodels/visibility/vis_utils.py:33: in calculate_transit_time
#     return site.target_meridian_transit_time(
# /opt/view/lib/python3.10/site-packages/astroplan/observer.py:1132: in target_meridian_transit_time
#     return self._determine_which_event(self._calc_transit,
# /opt/view/lib/python3.10/site-packages/astroplan/observer.py:917: in _determine_which_event
#     next_event = event_function('next')
# /opt/view/lib/python3.10/site-packages/astroplan/observer.py:906: in event_function
#     return function(time, target, w, antitransit=antitransit,
# /opt/view/lib/python3.10/site-packages/astroplan/observer.py:866: in _calc_transit
#     altaz = self.altaz(times, target, grid_times_targets=grid_times_targets)
# /opt/view/lib/python3.10/site-packages/astroplan/observer.py:497: in altaz
#     return target.transform_to(altaz_frame)
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/sky_coordinate.py:692: in transform_to
#     new_coord = trans(self.frame, generic_frame)
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/transformations.py:1588: in __call__
#     curr_coord = t(curr_coord, curr_toframe)
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/transformations.py:1154: in __call__
#     return supcall(fromcoord, toframe)
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/builtin_frames/icrs_observed_transforms.py:33: in icrs_to_observed
#     astrom = erfa_astrom.get().apco(observed_frame)
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/erfa_astrom.py:57: in apco
#     earth_pv, earth_heliocentric = prepare_earth_position_vel(obstime)
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/builtin_frames/utils.py:364: in prepare_earth_position_vel
#     jd1, jd2 = get_jd12(time, "tdb")
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/builtin_frames/utils.py:115: in get_jd12
#     newtime = getattr(time, scale)
# /opt/view/lib/python3.10/site-packages/astropy/time/core.py:1647: in __getattr__
#     tm._set_scale(attr)
# /opt/view/lib/python3.10/site-packages/astropy/time/core.py:769: in _set_scale
#     args.append(get_dt(jd1, jd2))
# /opt/view/lib/python3.10/site-packages/astropy/time/core.py:2490: in _get_delta_tdb_tt
#     self._delta_tdb_tt = erfa.dtdb(jd1, jd2, ut, 0.0, 0.0, 0.0)
# Karabo-Pipeline/karabo/test/test_000_astropy_env.py:32: in _dtdb_safe
#     return _orig_dtdb(date1, date2, ut, elong, u, v)
# Karabo-Pipeline/karabo/test/conftest.py:36: in _dtdb_safe
#     return _orig_dtdb(date1, date2, ut, elong, u, v)
# /opt/view/lib/python3.10/site-packages/erfa/core.py:16572: in dtdb
#     c_retval = ufunc.dtdb(date1, date2, ut, elong, u, v)
# E   ValueError: Invalid data-type for array
```

also

```txt
distributed 2022.10.2 requires tornado<6.2,>=6.0.3, but you have tornado 6.5.2 which is incompatible.
```

bonus:
7. make it possible to pip install --user as the jovyan user
8. it would be nice to not need two versions of cuda, you could investigate if oskar really needs to use cuda 10.
9. use build cache to speed up spack, requires the apt package called file

```dockerfile
RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    mkdir -p /opt/{software,view,buildcache,spack-source-cache,spack-misc-cache}; \
    spack env create --dir /opt/spack_env; \
    spack env activate /opt/spack_env; \
    spack config add "config:install_tree:root:/opt/software"; \
    spack config add "concretizer:unify:when_possible"; \
    spack config add "view:/opt/view"; \
    spack config add "config:source_cache:/opt/spack-source-cache"; \
    spack config add "config:misc_cache:/opt/spack-misc-cache"; \
    spack mirror add --autopush --unsigned mycache file:///opt/buildcache; \
    spack add \
        ...
    && \
    spack concretize --force && \
    spack install --no-check-signature --fail-fast \
    spack env view regenerate
```

but don't use `https://binaries.spack.io/develop`, because spack v0.23 is out of date.

## Verification

  • Build:
    • docker build -f sp5505.Dockerfile -t d3vnull0/sp5505:latest --progress=plain --push .
  • Confirm env activation and package import:
    • docker run --rm -it d3vnull0/sp5505:latest bash -lc "which python && python -c 'import karabo, sys; print(\"ok\", sys.version)'"
  • Run tests (again; they run at build too):
    • docker run --rm --gpus all d3vnull0/sp5505:latest bash -c 'pytest -x /home/jovyan/Karabo-Pipeline'

keep track of the size of the container

## gotchas / things to watch out for:

- spack specs have to be in quotes because they contain special characters like @
- DO NOT REPLACE `@=` WITH `@`
- consider swapping spack packages that are duplicates of apt installed packages as externals
- avoid conda, it doesn't play nicely with spack
- uninstall PyPI `argparse` whenever Spack CLI is used or before legacy source builds (e.g., python-casacore). The PyPI package shadows stdlib and breaks Spack’s argparse (`_SubParsersAction(required=...)`).
- `reproject` must be <0.10 for `rascil==1.0.0`. Pin to 0.9.1, install with `--no-deps`, and add `astropy-healpix==1.0.0` explicitly to avoid pulling `numpy>=1.24`.
- Build `python-casacore==3.5.0` from source against Spack `casacore` (use `+python` variant). Export `CMAKE_PREFIX_PATH`, `CASACORE_ROOT`, `CPATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH` to the Spack view and use `--no-binary=:all:`. Wheels can cause MS-reading segfaults.
- Keep core pins `numpy`, `scipy`, `pandas`, `xarray` the same between spack and pip. Use `--no-deps` on targeted installs (e.g., SKA packages, reproject) to prevent resolver upgrades.
- `xarray==2022.12.0` isn’t available in some Spack repos; use a Spack overlay repo and add it via `spack repo add /opt/karabo-spack`.
- For SCM-tagged packages (astroplan, photutils), set `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_*` and preinstall `astropy==5.1`. Use `--no-deps` where appropriate to avoid unintended upgrades. Otherwise you may get `0.0.0` versions or API errors.
- PyBDSF: prefer the wheel `bdsf==1.12.0`. If building from source, ensure `scikit-build` is present and set `SETUPTOOLS_USE_DISTUTILS=stdlib` plus `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BDSF`. Older builds can fail in f2py/metadata steps.
- OSKAR: run `cmake/make` with `LD_LIBRARY_PATH` preferring system libs to avoid `libcurl.so.4: no version information available` noise.
- Import names matter in probes: use `astropy_healpix` (underscore) and try both `aratmospy` and `ARatmospy` to validate installs without triggering resolver changes.

please don't remove my tests for target software versions

### scm versioning:

Some packages (astroplan, photutils) use setuptools_scm to manage versioning

# this does not work on earlier setuptools versions:
```bash
SETUPTOOLS_SCM_PRETEND_VERSION=${ASTROPLAN_VERSION} python -m pip install --no-build-isolation 'git+https://github.com/astropy/astroplan.git@v'${ASTROPLAN_VERSION}
SETUPTOOLS_SCM_PRETEND_VERSION=${ASTROPLAN_VERSION} python -m build --wheel --no-isolation && python -m pip install --no-build-isolation dist/astroplan-${ASTROPLAN_VERSION}-py3-none-any.whl
```
but this works:
```bash
git fetch --tags --depth=1 origin v${ASTROPLAN_VERSION} && python -m pip install --no-build-isolation .
```

I'm pretty sure this didn't work on an earlier setuptools version, but it works now:
```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTROPLAN=${ASTROPLAN_VERSION} &&python -m pip install --no-build-isolation 'git+https://github.com/astropy/astroplan.git@v'${ASTROPLAN_VERSION}
```

### specific package versions

some packages like rascil require super specific versions of packages not available in spack

```txt
edit ska-sdp-spack/packages/py-xarray/package.py and others from within the docker image

@package.py

the only xarray release that meets the requirements is @https://github.com/pydata/xarray/releases/tag/v2022.12.0

or failing that, create new entries in a new folder that can be mounted into the docker container and loaded as a spack repo under /opt/karabo-spack
```

