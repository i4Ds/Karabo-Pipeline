## Benchmark on {{ cluster }}

### Results (preliminary) 

#### Simulation Parameters
* Sky model {{ sky_model }}
* Number of sources:            {{ num_sources }}
* Number of frequency channels: {{ num_channels }}
* One run for each variation of frequency channel number

#### Computing Times
![Computing Times]({{ compute_image }})

#### File Sizes
![File Sizes]({{ size_image }})

The final size of the FITS files. The sizes are calculated using the `du -h` command in the output directory.


### Batch Script
```
{% for line in script %}{{ line }}{% endfor %}
```
If running the GPU benchmark I also set
```
#SBATCH --gpus=1
```
