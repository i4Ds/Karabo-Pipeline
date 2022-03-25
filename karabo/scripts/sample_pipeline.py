def run(telescope_config):
    import oskar
    import numpy as np
    from rascil.apps import rascil_imager

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
            "use_gpus": "False"
        },
        "observation": {
            "num_channels": "64",
            "start_frequency_hz": "100e6",
            "frequency_inc_hz": "20e6",
            "phase_centre_ra_deg": "20",
            "phase_centre_dec_deg": "-30",
            "num_time_steps": "24",
            "start_time_utc": "01-01-2000 12:00:00.000",
            "length": "12:00:00.000"
        },
        "telescope.png": {
            "input_directory": telescope_config
        },
        "interferometer": {
            "ms_filename": "visibilities.ms",
            "channel_bandwidth_hz": "1e6",
            "time_average_sec": "10"
        }
    }
    settings = oskar.SettingsTree("oskar_sim_interferometer")
    settings.from_dict(params)

    if precision == "single":
        settings["simulator/double_precision"] = "False"

    # Set the sky model and run the simulation.
    sim = oskar.Interferometer(settings=settings)
    sim.set_sky_model(sky)
    sim.run()

    def start_imager(rawargs):
        parser = rascil_imager.cli_parser()
        args = parser.parse_args(rawargs)
        rascil_imager.performance_environment(args.performance_file, mode="w")
        rascil_imager.performance_store_dict(args.performance_file, "cli_args", vars(args), mode="a")
        image_name = rascil_imager.imager(args)

    start_imager([
        '--ingest_msname', 'visibilities.ms',
        '--ingest_dd', '0',
        '--ingest_vis_nchan', '64',
        '--ingest_chan_per_blockvis', '4',
        '--ingest_average_blockvis', 'True',
        '--imaging_npixel', '2048',
        '--imaging_cellsize', '3.878509448876288e-05',
        '--imaging_weighting', 'robust',
        '--imaging_robustness', '-0.5',
        '--clean_nmajor', '5',
        '--clean_algorithm', 'mmclean',
        '--clean_scales', '0', '6', '10', '30', '60',
        '--clean_fractional_threshold', '0.3',
        '--clean_threshold', '0.12e-3',
        '--clean_nmoment', '5',
        '--clean_psf_support', '640',
        '--clean_restored_output', 'integrated'])
