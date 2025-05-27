# SRCNet Workflow AA*

This workflow is to execute a Karabo workflow `karabo.workflows.SRCNet_SKA-MID_AAstar.py` as a k8s job. The script is configurable via environment variables set in the script (and therefore in the job-manifest). In addition, it's configurable which data-products are gonna be ingested into the datalake by script arguments. The other data-products which are produced by the workflow , but are not specified for ingestion, will get discarded (but still have to be able to fit into container ephemeral storage or dedicated mounted volume). An example of this is if you're just interested in a cleaned image, visibilities are still created as part of the simulation. Thus, we recommend to put temporary data-products into a dedicated PVC `tmp-pvc.yaml` (because ephemeral storage of a node might not be enough) and output data-products have to fit into ingestor PVC.

## Resources

The following calculations are just min-sizes of according data-products, because they just account for required data and don't take metadata or additional things like flags into account. This might be useful to estimate required resources for larger simulations.

Casa MS min-size can be estimated by: `num_measurements * num_bytes(dtype)`, where `num_measurements = num_channels * num_time_stamps * num_baselines * num_polarizations` and `num_bytes(dtype) = 8` for complex64.
`num_measurements` can be calculated using `karabo.simulation.visibility.Visibility.num_measurements`.

Min-size for a single image can be estimated by: `imaging_npixel^2 * num_channels * num_bytes(dtype)`, where `num_channels = 1` for this workflow and `num_bytes(dtype) = 8` for float64.

A larger simulation can easily take several hours or days (also depending if GPU is available). Therefore it is advised to start with a smaller simulation to estimate durations for a longer run on your device. Check pod-logs for current status.

**Note:** In case your node is not setup to have reserved resources for kubelet, the according node might get in a unhealthy state if the jobs or namespace resources are not limited, because a node then can't talk with the master nodes anymore.

## Setup

### OIDC Secret

Setup oidc-client-secret using `data-ingestor/secrets.yaml` before doing helm installation. This requires according setup (e.g. vault) first. Then check if it exists `kubectl get secret -n datalake-ingestion`.

### Install Ingestor via Helm

Assumes that data-ingestor secret exists.

The current `values.yaml` of the ingestor is based on helm release `0.1.1`. In case you want to look at the helm deployment, please look at the last commit of the according helm version before it got bumped to `0.1.2`. It's because during `0.1.1` a lot of breaking changes were introduced which makes the initial values.yaml of `0.1.1` git revision invalid.

```bash
helm repo add ska-src-dm-di-ingestor https://gitlab.com/api/v4/projects/51600992/packages/helm/stable
helm repo update
helm install ingestor ska-src-dm-di-ingestor/ska-src-dm-di-ingestor \
  --version 0.1.1 \
  --namespace datalake-ingestion \
  -f <path/to/your/values.yaml>
```

### Workflow Setup

Apply tmpdir for disk caching:

```bash
kubectl apply -f tmp-pvc.yaml
```

## Launch Job

Launch job with according parameters as environment variables.

To have unique data-products in rucio, you may want to set some variables differently for each run:
- `OBS_ID`: obscore observation id
- `FILE_PREFIX`: file prefix to have different file-names for all resulting data products.

The resulting data-products are specified as flags in the command args of the job manifest. Valid options are: `--dirty`, `--cleaned` and `--visibility`. But be aware that a datalake might not support ingestion of hierarchical data-products like casa MSv2 visibilities.

```bash
kubectl apply -f job.yaml
```

In case of running multiple or consecutive jobs, deleting completed jobs or rename job-name may be required.

Alternatively, there is a CronJob version of this job that will trigger once every hour on the hour. It uses timestamps to disambiguate the fields that need to be unique otherwise.

```bash
kubectl apply -f cronjob.yaml
```


## Verify Job

To look if a job run successful, look at job pod logs (e.g. using `k9s` or `kubectl logs <pod-name> -n datalake-ingestion`).

To verify whether according data-products with it's according `.meta` metadata got ingested into the datalake, look into according ingestor directories. Failed ingestions are found in `/tmp/ingest/failed_processing/` where successful ingestions are in `/tmp/ingest/processed`. Just exec into the pod to look up the status:

```bash
kubectl exec -it core-<xxxxxxxxxx>-<xxxxx> -n datalake-ingestion -- bash
```
