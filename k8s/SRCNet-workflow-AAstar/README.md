# SRCNet Workflow AA*

This workflow is to execute a Karabo workflow `karabo.workflows.SRCNet_AAstar.py` as a k8s job. The script is configurable via environment variables set in the script (and therefore in the job-manifest). In addition, it's configurable which data-products are gonna be ingested into the datalake by script arguments. The other data-products which were produced by the workflow , but are not specified for ingestion, will get discarded (but still have to be able to fit into container ephemeral storage). An example of this is if you're just interested in a cleaned image, visibilities are still created as part of the simulation. Thus, intermediate data-products have to fit into karabo container ephemeral storage and output data-products have to fit into ingestor PVC.

## Resources

The following calculations are just min-sizes of according data-products, because they just account for required data and don't take metadata or additional things like Flags into account. This might be useful to estimate required resources for larger simulations.

Casa MS min-size can be estimated by: `num_measurements * num_bytes(dtype)`, where `num_measurements = num_channels * num_time_stamps * num_baselines * num_polarizations` and `num_bytes(dtype) = 8` for complex64.
`num_measurements` can be calculated using `karabo.simulation.visibility.Visibility.num_measurements`.

Image min-size for a single can be estimated by: `imaging_npixel^2 * num_channels * num_bytes(dtype)`, where `num_channels = 1` for this workflow and `num_bytes(dtype) = 8` for float64.

A large simulation can take several hours or days (also depending if GPU is available). Therefore it is advised to start with a smaller simulation to estimate durations for a longer run on your device. Check pod-logs for current status.

## Setup

### Install Ingestor via Helm
```bash
helm repo add ska-src-dm-di-ingestor https://gitlab.com/api/v4/projects/51600992/packages/helm/stable
helm repo update
helm install ingestor ska-src-dm-di-ingestor/ska-src-dm-di-ingestor \
  --version 0.1.1 \
  --namespace datalake-ingestion \
  --create-namespace \
  -f <path/to/your/values.yaml>  # update path to values.yaml
```