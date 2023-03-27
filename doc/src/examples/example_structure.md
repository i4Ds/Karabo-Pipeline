# Examples

## Running an interferometer simulation

Running an interferometer simulation is really easy.
Please look at the karabo.package documentation for specifics on the individual functions.

```python
<example_interfe_simu.py>
```

## Show telescope config

```python
<example_tel_set.py>
```

![Image](../images/telescope.png)

## Use Karabo on a SLURM cluster
With our helper function `setup_dask_on_slurm` you can easily setup a dask cluster on a SLURM cluster. 
This client can then be passed to functions supporting dask clients to parallise the computation to different nodes.

```python
<example_setup_dask_on_slurm.py>
```
Use the client to run a simulation

```python
deconvolved, restored, residual = imager_askap.imaging_rascil(
    ...
    client=client,
    ...
)
```
