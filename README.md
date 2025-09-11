# dnplot 💻

**dnplot** is a tool designed for the visualization of [**dnora**](https://github.com/MET-OM/dnora) objects, such as 🌬️ wind, 🌊 wave and ocean current data.

![dnplot Visualization](https://github.com/bjorkqvi/dnplot/blob/doc/docs/files/spectra_plotly.gif)


## Quick Installation 🚀 

Ready to dive in? You can easily install **dnplot** using pip:

```shell
$ pip install dnplot 
```

## Example

To quickly visualize the wave data outputted by SWAN to a netcdf file:

```python
import dnplot
from dnora.wavegrid import WaveGrid

data = WaveGrid.from_netcdf('model_output.nc')
point = data.sel(lon=5.25, lat=62.3, method='nearest') # Pick just one point

# The dict could be a dnora ModelRun object containing the data
plot = dnplot.Matplotlib({'wavegrid': data, 'waveseries': point})

# default value for var=["hs", ("tm01", "tm02"), "dirm"] 
plot.waveseries(var=['hs', ('tp','tm01'), 'dirp']) 
plot.wavegrid('hs')
```

## Docs 📚
Looking for more details? Our documentation (currently under development) provides information about the package, its features, and how to use it. Explore it [here](https://dnora.readthedocs.io/en/latest/index.html).
