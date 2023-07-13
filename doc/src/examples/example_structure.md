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

Karabo manages all available nodes through Dask, making the computational power conveniently accessible for the user. The `DaskHandler` class streamlines the creation of a Dask client and offers a user-friendly interface for interaction. This class contains static variables, which when altered, modify the behavior of the Dask client. 

While users are not required to interact with Dask directly - thanks to the background processes managed by Karabo - the Dask client has to be initialized at the beginning of your script with `DaskHandler.setup` (see example below). This has to do with the spawning of new processes when creating `Nanny` processes.

If you need the client yourself, then no `setup()` is needed.

```python
from karabo.util.dask import DaskHandler

if __name__ == "__main__":
    # Get the Dask client
    client = DaskHandler.get_dask_client() # Not needed anymore to call .setup()

    # Use the client as needed
    result = client.submit(my_function, *args)
```
```python
from karabo.util.dask import DaskHandler

if __name__ == "__main__":
    DaskHandler.setup()
    result = <function_of_karabo_which_uses_dask_in_the_background>(*args)
```

Disable the usage of Dask by Karabo.

```python
from karabo.util.dask import DaskHandler
# Modify the static variables
DaskHandler.use_dask = False
```

Please also check out the `DaskHandler` under `karabo.util.dask` for more information.

### Dask Dashboard
The link for the Dask Dashboard is written into a .txt file called `karabo-dask-dashboard.txt`. This file is located in the same directory as where the run was started. This URL can then be pasted into a browser to access the Dask Dashboard. If you run Karabo on a VM without access to a browser and internet, you can use `port forwarding` to access the Dask Dashboard from your local machine. In `VSCODE`, this can be done directly when using the "PORTS" tab; just paste the IP address and port number from the .txt file into the Port column and click on "Open in Browser" in the Local Adress column.
