def run():
    import oskar
    import numpy as np

    from rascil.workflows import \
        invert_list_rsexecute_workflow, \
        deconvolve_list_rsexecute_workflow, \
        create_blockvisibility_from_ms_rsexecute, rsexecute, \
        weight_list_rsexecute_workflow, \
        continuum_imaging_skymodel_list_rsexecute_workflow

    from rascil.workflows.rsexecute.execution_support import rsexecute
    from rascil.processing_components import create_image_from_visibility
    from rascil.processing_components.visibility.operations import convert_blockvisibility_to_stokesI
    from rascil.data_models import PolarisationFrame

    # Set the numerical precision to use.
    precision = "single"

    # Create a sky model containing three sources from a numpy array.
    sky_data = np.array([
        [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
        [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
        [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10]])
    sky = oskar.Sky.from_array(sky_data, precision)  # Pass precision here.

    params = {
        "simulator": {
            "use_gpus": False
        },
        "observation": {
            "num_channels": 64,
            "start_frequency_hz": 100e6,
            "frequency_inc_hz": 20e6,
            "phase_centre_ra_deg": 20,
            "phase_centre_dec_deg": -30,
            "num_time_steps": 24,
            "start_time_utc": "01-01-2000 12:00:00.000",
            "length": "12:00:00.000"
        },
        "telescope": {
            "input_directory": "../data/telescope.tm"
        },
        "interferometer": {
            "ms_filename": "visibilities.ms",
            "channel_bandwidth_hz": 1e6,
            "time_average_sec": 10
        }
    }
    settings = oskar.SettingsTree("oskar_sim_interferometer")
    settings.from_dict(params)

    if precision == "single":
        settings["simulator/double_precision"] = False

    # Set the sky model and run the simulation.
    sim = oskar.Interferometer(settings=settings)
    sim.set_sky_model(sky)
    sim.run()

    dds = [0]
    channels_per_dd = 64
    nchan_per_blockvis = 4
    nout = channels_per_dd // nchan_per_blockvis

    # Create a list of blockvisibilities
    bvis_list = create_blockvisibility_from_ms_rsexecute('visibilities.ms/',
                                                         nchan_per_blockvis=nchan_per_blockvis,
                                                         dds=dds,
                                                         nout=nout,
                                                         average_channels=True)

    bvis_list = [rsexecute.execute(convert_blockvisibility_to_stokesI)(vis) for vis in bvis_list]

    # create model images from all visibilites
    modelimage_list = [rsexecute.execute(create_image_from_visibility)(vis,
                                                                       npixel=2048,
                                                                       nchan=1,
                                                                       cellsize=3.878509448876288e-05,
                                                                       polarisationFrame=PolarisationFrame('stokesI'))
                       for vis in bvis_list]

    # weight visibilities
    bvis_list = weight_list_rsexecute_workflow(bvis_list,
                                               modelimage_list,
                                               weigthing='robust',
                                               robustness=-0.5)

    result = continuum_imaging_skymodel_list_rsexecute_workflow(
        bvis_list,
        modelimage_list,
        context='ng',
        threads=4,
        wstacking=True,
        niter=1000,
        nmajor=5,
        algorithm='mmclean',
        gain=0.1,
        scales=[0, 6, 10, 30, 60],
        fractional_threshold=0.3,
        threshold=0.00012,
        nmoment=5,
        psf_support=640,
        restored_output='integrated',
        deconvolve_facets=1,
        deconvolve_overlap=32,
        deconvolve_taper='tukey',
        restore_facets=1,
        restore_overlap=32,
        restore_taper='tukey',
        dft_compute_kernel=None,
        component_threshold=None,
        component_method='fit',
        flat_sky=False,
        clean_beam=None,
    )

    # start computation on dask cluster
    result = rsexecute.compute(result, sync=True)

