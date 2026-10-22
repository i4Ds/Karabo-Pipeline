Setup:
- Create and activate a Python 3.10 (3.11 should also work, not tested) virtual env / conda env / similar
- pip install ska-ost-array-config --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple

array_config_to_karabo_json.py (SKA-Mid telescope JSON with dish SEFDs):
- Tested with Python 3.12, ska-ost-array-config 3.2.0 and ska-ost-senscalc 12.0.1
- pip install ska-ost-array-config==3.2.0 ska-ost-senscalc==12.0.1 --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple
- Run from karabo/data: python _add_oskar_ska_layouts/array_config_to_karabo_json.py
- ska-ost-array-config 3.2.0 reproduces the AA* positions of SKA-MID-AAstar.ska-ost-array-config-2.3.1.tm exactly, which is no longer available on the package index. Later versions (e.g. 5.0.0) change the dish selection and order.
