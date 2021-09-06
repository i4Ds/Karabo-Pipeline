""" RASCIL Continuum Imaging

"""

import argparse
import datetime
import logging
import os
import pprint
import time

import matplotlib

matplotlib.use("Agg")

from distributed import Client, SSHCluster
import dask

from rascil.data_models import PolarisationFrame, export_skymodel_to_hdf5
from rascil.processing_components.util.sizeof import get_size

from rascil.processing_components import (
    create_image_from_visibility,
    qa_image,
    convert_blockvisibility_to_stokesI,
    export_image_to_fits,
    image_gather_channels,
    create_calibration_controls,
    advise_wide_field,
)

from rascil.processing_components.util.performance import (
    performance_blockvisibility,
    performance_store_dict,
    performance_qa_image,
    performance_dask_configuration,
    performance_environment,
)

from rascil.workflows import (
    weight_list_rsexecute_workflow,
    continuum_imaging_skymodel_list_rsexecute_workflow,
    remove_sumwt,
    create_blockvisibility_from_ms_rsexecute,
    ical_skymodel_list_rsexecute_workflow,
    invert_list_rsexecute_workflow,
    sum_invert_results_rsexecute,
    taper_list_rsexecute_workflow,
)

from rascil.workflows.rsexecute.execution_support.rsexecute import (
    rsexecute,
    get_dask_client,
)

from rascil.apps.apps_parser import (
    apps_parser_imaging,
    apps_parser_cleaning,
    apps_parser_dask,
    apps_parser_ingest,
    apps_parser_app,
    apps_parser_calibration,
)

log = logging.getLogger("rascil-logger")
log.setLevel(logging.INFO)


def cli_parser():
    """Get a command line parser and populate it with arguments

    First a CLI argument parser is created. Each function call adds more arguments to the parser.

    :return: CLI parser argparse
    """

    parser = argparse.ArgumentParser(
        description="RASCIL continuum imager", fromfile_prefix_chars="@"
    )
    parser = apps_parser_app(parser)
    parser = apps_parser_ingest(parser)
    parser = apps_parser_imaging(parser)
    parser = apps_parser_calibration(parser)
    parser = apps_parser_cleaning(parser)
    parser = apps_parser_dask(parser)

    return parser


def imager(args):
    """Continuum imager

    The return contains names of the files written to disk as fits files.

    mode=invert: dirty image
    mode=cip: deconvolved image, residual image, restored image.
    mode=ical: deconvolved image, residual image, restored image
    mode=load: load and list the data

    :param args: argparse with appropriate arguments
    :return: Names of outputs as fits files
    """

    # We need to tell all the Dask workers to use the same log
    cwd = os.getcwd()

    assert args.ingest_msname is not None, "Input msname must be specified"

    if args.logfile is None:
        logfile = args.ingest_msname.replace(".ms", ".log")
    else:
        logfile = args.logfile

    def init_logging():
        logging.basicConfig(
            filename=logfile,
            filemode="a",
            format="%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%d/%m/%Y %I:%M:%S %p",
            level=logging.INFO,
        )

    init_logging()

    log.info("\nRASCIL Continuum Imager\n")

    starttime = datetime.datetime.now()
    log.info("Started : {}".format(starttime))
    log.info("Writing log to {}".format(logfile))

    setup_rsexecute(args)

    rsexecute.run(init_logging)

    log.info(pprint.pformat(vars(args)))

    log.info("Current working directory is {}".format(cwd))

    bvis_list, msname = get_blockvis_list(args)

    # Now get the blockvisibility info. Do the query on the cluster to avoid
    # transferring the entire data.
    perf_save_blockvis_info(args, bvis_list)

    # If the cellsize has not been specified, we compute the blockvis now and
    # run the advisor
    cellsize = get_cellsize(args, bvis_list)

    # Make only the Stokes I image so we convert the visibility to Stokes I
    bvis_list = convert_to_stokesI(args, bvis_list)

    npixel = args.imaging_npixel

    # Define the model to be used as a template, one for each BlockVisibility
    model_list = create_model_image_list(args, bvis_list, cellsize, npixel)

    bvis_list = weight_blockvis(args, bvis_list, model_list)

    clean_beam = get_clean_beam(args)

    # Now the actual processing
    if args.mode == "cip":
        results = cip(args, bvis_list, model_list, msname, clean_beam)
    elif args.mode == "ical":
        results = ical(args, bvis_list, model_list, msname, clean_beam)
    elif args.mode == "invert":
        results = invert(args, bvis_list, model_list, msname)
    elif args.mode == "load":
        results = load(args, bvis_list, msname)
    else:
        raise ValueError("Unknown mode {}".format(args.mode))

    # Save the processing statistics from Dask
    dask_info = rsexecute.save_statistics(logfile.replace(".log", ""))

    perf_save_dask_profile(args, dask_info)

    rsexecute.close()

    log.info("Resulting image(s) {}".format(results))

    log.info("Started  : {}".format(starttime))
    log.info("Finished : {}".format(datetime.datetime.now()))

    return results


def setup_rsexecute(args):
    # We can run distributed (use_dask=True) or in serial (use_dask=False). Using Dask is usually recommended
    if args.use_dask == "True":
        if args.dask_scheduler == "existing":
            log.info("Using existing dask client")
        elif args.dask_scheduler == "ssh":
            log.info("Using SSH scheduler")
            cluster = SSHCluster(
                args.dask_nodes,
                connect_options={"known_hosts": None},
                worker_options={"nthreads": args.nthreads},
                scheduler_options={"port": 0, "dashboard_address": ":8787"},
            )
            client = Client(cluster)
            rsexecute.set_client(use_dask=True, client=client)
        elif args.dask_scheduler is not None:
            log.info("Using specified scheduler {}".format(args.dask_scheduler))
            client = Client(address=args.dask_scheduler)
            rsexecute.set_client(use_dask=True, client=client)
        elif args.dask_scheduler_file is not None:
            log.info(
                "Using specified scheduler file {}".format(args.dask_scheduler_file)
            )
            client = Client(scheduler_file=args.dask_scheduler_file)
            rsexecute.set_client(use_dask=True, client=client)
        else:
            log.info("Gettting client via get_dask_client")
            client = get_dask_client(
                n_workers=args.dask_nworkers,
                threads_per_worker=args.dask_nthreads,
                memory_limit=args.dask_memory,
            )
            rsexecute.set_client(use_dask=True, client=client)

        rsexecute.init_statistics()
        # Sample the memory usage with a scheduler plugin
        if rsexecute.using_dask and args.dask_memory_usage_file is not None:
            rsexecute.memusage(args.dask_memory_usage_file)
        if args.dask_tcp_timeout is not None:
            dask.config.set({"distributed.comm.timeouts.tcp": args.dask_tcp_timeout})

        if args.dask_connect_timeout is not None:
            dask.config.set(
                {"distributed.comm.timeouts.tcp": args.dask_connect_timeout}
            )

        tcp_timeout = dask.config.get("distributed.comm.timeouts.tcp")
        connect_timeout = dask.config.get("distributed.comm.timeouts.connect")

        log.info(f"Dask timeouts: connect {connect_timeout} tcp: {tcp_timeout}")
        performance_dask_configuration(args.performance_file, rsexecute)
    else:
        rsexecute.set_client(use_dask=False)


def get_blockvis_list(args):
    # Read in the MS into a list of BlockVisibility's
    # We start with an MS with e.g. 4 data_descriptors, each of which has e.g. 64 channels.
    # We average each dd over e.g. 2 blocks of e.g. 32 channels, giving e.g. 8 separate
    # BlockVisibility's
    msname = args.ingest_msname
    dds = args.ingest_dd
    channels_per_dd = args.ingest_vis_nchan
    nchan_per_blockvis = args.ingest_chan_per_blockvis
    nout = channels_per_dd // nchan_per_blockvis
    bvis_list = create_blockvisibility_from_ms_rsexecute(
        msname=args.ingest_msname,
        dds=dds,
        nout=nout,
        nchan_per_blockvis=nchan_per_blockvis,
        average_channels=args.ingest_average_blockvis == "True",
    )
    return bvis_list, msname


def perf_save_dask_profile(args, dask_info):
    if args is not None and args.performance_file is not None:
        performance_store_dict(
            args.performance_file, "dask_profile", dask_info, mode="a"
        )
        performance_dask_configuration(args.performance_file, rsexecute, mode="a")


def get_clean_beam(args):
    if args.clean_beam is not None:
        clean_beam = {
            "bmaj": args.clean_beam[0],
            "bmin": args.clean_beam[1],
            "bpa": args.clean_beam[2],
        }
    else:
        clean_beam = None
    return clean_beam


def weight_blockvis(args, bvis_list, model_list):
    # Create a graph to weight the data
    if args.imaging_weighting != "natural":
        bvis_list = weight_list_rsexecute_workflow(
            bvis_list,
            model_list,
            weighting=args.imaging_weighting,
            robustness=args.imaging_robustness,
        )
    if args.imaging_gaussian_taper is not None:
        bvis_list = taper_list_rsexecute_workflow(
            bvis_list, args.imaging_gaussian_taper
        )
    return bvis_list


def create_model_image_list(args, bvis_list, cellsize, npixel):
    model_list = [
        rsexecute.execute(create_image_from_visibility)(
            bvis,
            npixel=npixel,
            nchan=args.imaging_nchan,
            cellsize=cellsize,
            polarisation_frame=PolarisationFrame(args.imaging_pol),
        )
        for bvis in bvis_list
    ]
    return model_list


def convert_to_stokesI(args, bvis_list):
    if args.imaging_pol == "stokesI":
        bvis_list = [
            rsexecute.execute(convert_blockvisibility_to_stokesI)(bv)
            for bv in bvis_list
        ]
    return bvis_list


def get_cellsize(args, bvis_list):
    cellsize = args.imaging_cellsize
    if cellsize is None:
        advice_list = [
            rsexecute.execute(advise_wide_field)(bv, guard_band_image=3.0)
            for bv in bvis_list
        ]
        advice_list = rsexecute.compute(advice_list, sync=True)
        cellsize = advice_list[0]["cellsize"]
        log.info(f"Setting cellsize to {cellsize} rad")
    return cellsize


def perf_save_blockvis_info(args, bvis_list):
    if args.performance_file is not None:
        results = [
            rsexecute.execute(performance_blockvisibility)(bvis) for bvis in bvis_list
        ]
        results = rsexecute.compute(results, sync=True)
        for ibvis, r in enumerate(results):
            performance_store_dict(
                args.performance_file, f"blockvis{ibvis}", r, mode="a"
            )


def load(args, bvis_list, msname):
    """Load MS data

    :param args: The parameters read from the CLI using argparse
    :param bvis_list: A list of or graph to make BlockVisibilitys
    :param msname: The filename of the MeasurementSet
    :return: msname
    """
    bvis_list = rsexecute.compute(bvis_list, sync=True)
    for bvis in bvis_list:
        log.info(str(bvis))
    return msname


def cip(args, bvis_list, model_list, msname, clean_beam=None):
    """Run continuum imaging pipeline

    :param args: The parameters read from the CLI using argparse
    :param bvis_list: A list of or graph to make BlockVisibilitys
    :param model_list: A list of or graph to make model images
    :param msname: The filename of the MeasurementSet
    :param clean_beam: None or dict e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    :return: Names of output images (deconvolved, residual, restored)
    """
    start = time.time()
    result = continuum_imaging_skymodel_list_rsexecute_workflow(
        bvis_list,  # List of BlockVisibilitys
        model_list,  # List of model images
        context=args.imaging_context,  # Use nifty-gridder
        threads=args.imaging_ng_threads,
        wstacking=args.imaging_w_stacking == "True",  # Correct for w term in gridding
        niter=args.clean_niter,  # iterations in minor cycle
        nmajor=args.clean_nmajor,  # Number of major cycles
        algorithm=args.clean_algorithm,
        gain=args.clean_gain,  # CLEAN loop gain
        scales=args.clean_scales,  # Scales for multi-scale cleaning
        fractional_threshold=args.clean_fractional_threshold,
        # Threshold per major cycle
        threshold=args.clean_threshold,  # Final stopping threshold
        nmoment=args.clean_nmoment,
        # Number of frequency moments (1 = no dependence)
        psf_support=args.clean_psf_support,
        # Support of PSF used in minor cycles (halfwidth in pixels)
        restored_output=args.clean_restored_output,  # Type of restored image
        deconvolve_facets=args.clean_facets,
        deconvolve_overlap=args.clean_overlap,
        deconvolve_taper=args.clean_taper,
        restore_facets=args.clean_restore_facets,
        restore_overlap=args.clean_restore_overlap,
        restore_taper=args.clean_restore_taper,
        dft_compute_kernel=args.imaging_dft_kernel,
        component_threshold=args.clean_component_threshold,
        component_method=args.clean_component_method,
        flat_sky=args.imaging_flat_sky,
        clean_beam=clean_beam,
    )
    perf_graph(result, "cip", start, args.performance_file)
    # Execute the Dask graph
    log.info("rascil.imager.cip: Starting compute of continuum imaging pipeline graph ")
    result = rsexecute.compute(result, sync=True)
    log.info("rascil.imager.cip: Finished compute of continuum imaging pipeline graph")

    imagename = msname.replace(".ms", "_nmoment{}_cip".format(args.clean_nmoment))
    return write_results(
        args.clean_restored_output, imagename, result, args.performance_file
    )


def perf_graph(result, name, start, performance_file):
    duration = time.time() - start
    size = get_size(result)
    graph = {"name": name, "time": duration, "size": size}
    log.info(
        f"rascil.imager.perf_graph: Size of {name} graph = {get_size(result)} B, time to construct {duration} s"
    )
    performance_store_dict(performance_file, "graph", graph)


def write_results(restored_output, imagename, result, performance_file):
    """Write the results out to files

    :param restored_output: Type of output: list or taylor
    :param imagename: Root of image names
    :param result: Set of results i.e. deconvolved, residual, restored, skymodel
    :param performance_file: Name of performance file
    :return:
    """
    residual, restored, skymodel = result

    deconvolvedname = None
    residualname = None
    restoredname = None

    deconvolved = [sm.image for sm in skymodel]
    skymodelname = imagename + "_skymodel.hdf"
    export_skymodel_to_hdf5(skymodel, skymodelname)
    del skymodel

    if restored_output == "list":
        deconvolved_image = image_gather_channels(deconvolved)
        del deconvolved
        performance_qa_image(
            performance_file, "deconvolved", deconvolved_image, mode="a"
        )
        log.info(qa_image(deconvolved_image, context="Deconvolved"))
        deconvolvedname = imagename + "_deconvolved.fits"
        export_image_to_fits(deconvolved_image, deconvolvedname)
        del deconvolved_image
        restored = image_gather_channels(restored)
        performance_qa_image(performance_file, "restored", restored, mode="a")
        log.info("Writing restored image as spectral cube")
        restoredname = imagename + "_restored.fits"
        export_image_to_fits(restored, restoredname)

        residual = remove_sumwt(residual)
        residual_image = image_gather_channels(residual)
        del residual
        performance_qa_image(performance_file, "residual", residual_image, mode="a")
        log.info("Writing residual image as spectral cube")
        log.info(qa_image(residual_image, context="Residual"))
        residualname = imagename + "_residual.fits"
        export_image_to_fits(residual_image, residualname)
        del residual_image

    elif restored_output == "taylor":
        nmoment = len(restored)
        # Do the first last so that the name will be correct for Taylor 0
        for taylor in range(nmoment - 1, -1, -1):
            performance_qa_image(
                performance_file,
                f"deconvolved taylor{taylor}",
                deconvolved[taylor],
                mode="a",
            )
            log.info(
                qa_image(deconvolved[taylor], context=f"Deconvolved taylor{taylor}")
            )
            deconvolvedname = imagename + f".taylor.{taylor}.deconvolved.fits"
            export_image_to_fits(deconvolved[taylor], deconvolvedname)

            performance_qa_image(
                performance_file, f"restored taylor{taylor}", restored[taylor], mode="a"
            )
            log.info("Writing restored image")
            log.info(qa_image(restored[taylor], context=f"Restored taylor{taylor}"))
            restoredname = imagename + f".taylor.{taylor}.restored.fits"
            export_image_to_fits(restored[taylor], restoredname)

            performance_qa_image(
                performance_file,
                f"residual_taylor{taylor}",
                residual[taylor][0],
                mode="a",
            )
            log.info(qa_image(residual[taylor][0], context=f"Residual taylor{taylor}"))
            residualname = imagename + f".taylor.{taylor}.residual.fits"
            export_image_to_fits(residual[taylor][0], residualname)

    else:
        deconvolved_image = image_gather_channels(deconvolved)
        del deconvolved
        performance_qa_image(
            performance_file,
            "deconvolved",
            deconvolved_image,
            mode="a",
        )
        log.info(qa_image(deconvolved_image, context=f"Deconvolved"))
        deconvolvedname = imagename + f"_deconvolved.fits"
        export_image_to_fits(deconvolved_image, deconvolvedname)
        del deconvolved_image

        log.info("Writing restored image as single plane at mid-frequency")
        restoredname = imagename + "_restored_centre.fits"
        performance_qa_image(performance_file, "restored_centre", restored, mode="a")
        export_image_to_fits(restored, restoredname)

        log.info(qa_image(restored, context="Restored"))
        del restored

        residual = remove_sumwt(residual)
        residual_image = image_gather_channels(residual)
        del residual
        performance_qa_image(performance_file, "residual", residual_image, mode="a")
        log.info(qa_image(residual_image, context="Residual"))
        residualname = imagename + "_residual.fits"
        export_image_to_fits(residual_image, residualname)
        del residual_image

    return (deconvolvedname, residualname, restoredname, skymodelname)


def ical(args, bvis_list, model_list, msname, clean_beam=None):
    """Run ICAL pipeline

    :param args: The parameters read from the CLI using argparse
    :param bvis_list: A list of or graph to make BlockVisibilitys
    :param model_list: A list of or graph to make model images
    :param msname: The filename of the MeasurementSet
    :param clean_beam: None or dict e.g. {"bmaj":0.1, "bmin":0.05, "bpa":-60.0}. Units are deg, deg, deg
    :return: Names of output images (deconvolved, residual, restored)
    """
    controls = create_calibration_controls()

    controls["T"]["first_selfcal"] = args.calibration_T_first_selfcal
    controls["T"]["phase_only"] = args.calibration_T_phase_only
    controls["T"]["timeslice"] = args.calibration_T_timeslice

    controls["G"]["first_selfcal"] = args.calibration_G_first_selfcal
    controls["G"]["timeslice"] = args.calibration_G_timeslice

    controls["B"]["first_selfcal"] = args.calibration_B_first_selfcal
    if args.calibration_B_timeslice is None:
        controls["B"]["timeslice"] = 1e5
    else:
        controls["B"]["timeslice"] = args.calibration_B_timeslice

    # Next we define a graph to run the continuum imaging pipeline
    start = time.time()

    result = ical_skymodel_list_rsexecute_workflow(
        bvis_list,  # List of BlockVisibilitys
        model_list,  # List of model images
        context=args.imaging_context,  # Use nifty-gridder
        threads=args.imaging_ng_threads,
        wstacking=args.imaging_w_stacking == "True",  # Correct for w term in gridding
        niter=args.clean_niter,  # iterations in minor cycle
        nmajor=args.clean_nmajor,  # Number of major cycles
        algorithm=args.clean_algorithm,
        gain=args.clean_gain,  # CLEAN loop gain
        scales=args.clean_scales,  # Scales for multi-scale cleaning
        fractional_threshold=args.clean_fractional_threshold,
        # Threshold per major cycle
        threshold=args.clean_threshold,  # Final stopping threshold
        nmoment=args.clean_nmoment,
        # Number of frequency moments (1 = no dependence)
        psf_support=args.clean_psf_support,
        # Support of PSF used in minor cycles (halfwidth in pixels)
        restored_output=args.clean_restored_output,  # Type of restored image
        deconvolve_facets=args.clean_facets,
        deconvolve_overlap=args.clean_overlap,
        deconvolve_taper=args.clean_taper,
        restore_facets=args.clean_restore_facets,
        restore_overlap=args.clean_restore_overlap,
        restore_taper=args.clean_restore_taper,
        calibration_context=args.calibration_context,
        controls=controls,
        global_solution=args.calibration_global_solution,
        component_threshold=args.clean_component_threshold,
        component_method=args.clean_component_method,
        dft_compute_kernel=args.imaging_dft_kernel,
        clean_beam=clean_beam,
    )
    perf_graph(result, "ical", start, args.performance_file)

    # Execute the Dask graph
    log.info("rascil.imager.ical: Starting compute of ical pipeline graph ")
    residual, restored, skymodel, gt_list = rsexecute.compute(result, sync=True)
    log.info("rascil.imager.ical: Finished compute of ical pipeline graph")

    imagename = msname.replace(".ms", "_nmoment{}_ical".format(args.clean_nmoment))
    return write_results(
        args.clean_restored_output,
        imagename,
        (residual, restored, skymodel),
        args.performance_file,
    )


def invert(args, bvis_list, model_list, msname):
    """Run invert

    :param args: The parameters read from the CLI using argparse
    :param bvis_list: A list of or graph to make BlockVisibilitys
    :param model_list: A list of or graph to make model images
    :param msname: The filename of the MeasurementSet
    :return: Names of output image (dirty image or psf image)
    """
    # Next we define a graph to run the continuum imaging pipeline
    start = time.time()
    result = invert_list_rsexecute_workflow(
        bvis_list,  # List of BlockVisibilitys
        model_list,  # List of model images
        context=args.imaging_context,
        dopsf=args.imaging_dopsf == "True",
        threads=args.imaging_ng_threads,
        wstacking=args.imaging_w_stacking == "True",
        dft_compute_kernel=args.imaging_dft_kernel,
    )
    result = sum_invert_results_rsexecute(result)
    perf_graph(result, "invert", start, args.performance_file)
    # Execute the Dask graph
    log.info("rascil.imager.invert: Starting compute of invert graph ")
    dirty, sumwt = rsexecute.compute(result, sync=True)
    log.info("rascil.imager.invert: Finished compute of invert graph")
    imagename = msname.replace(".ms", "_invert")

    if args.imaging_dopsf == "True":
        performance_qa_image(args.performance_file, "psf", dirty, mode="a")
        log.info(qa_image(dirty, context="PSF"))
        psfname = imagename + "_psf.fits"
        export_image_to_fits(dirty, psfname)
        return psfname
    else:
        performance_qa_image(args.performance_file, "dirty", dirty, mode="a")
        log.info(qa_image(dirty, context="Dirty"))
        dirtyname = imagename + "_dirty.fits"
        export_image_to_fits(dirty, dirtyname)
        return dirtyname


def main(rawargs):
    # Get command line inputs
    parser = cli_parser()
    args = parser.parse_args(rawargs)
    performance_environment(args.performance_file, mode="w")
    performance_store_dict(args.performance_file, "cli_args", vars(args), mode="a")
    image_name = imager(args)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
