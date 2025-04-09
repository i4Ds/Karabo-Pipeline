# SRCNet Workflow AA*

This workflow is to execute a Karabo workflow `karabo.workflows.SRCNet_AAstar.py` as a k8s job. The script is configurable via environment variables set in the script (and therefore in the job-manifest). In addition, it's configurable which data-products are gonna be ingested into the datalake by script arguments. The other data-products which were produced by the workflow , but are not specified for ingestion, will get discarded (but still have to be able to fit into container ephemeral storage). An example of this is if you're just interested in a cleaned image, visibilities are still created as part of the simulation. Thus, intermediate data-products have to fit into karabo container ephemeral storage and output data-products have to fit into ingestor PVC.

Storage size to consider: TODO (there are functions there to estimate size)

Duration of workflow: TODO

## Setup

### Install Ingestor
```bash
helm repo add ska-src-dm-di-ingestor https://gitlab.com/api/v4/projects/51600992/packages/helm/stable
helm repo update
helm install ingestor ska-src-dm-di-ingestor/ska-src-dm-di-ingestor \
  --version 0.1.1 \
  --namespace <your-namespace> \  # update namespace
  --create-namespace \
  -f <path/to/your/values.yaml>  # update path to values.yaml
```