import dnora as dn
import dnplot


point = dn.grid.Grid(lon=4.308, lat=62.838, name="Svinoy")
model = dn.modelrun.NORA3(point, year=2022, month=3, day=18)
model.import_spectra()
model.spectra_to_waveseries()

plot = dnplot.Plotly(model)
plot.waveseries(use_dash=False)
