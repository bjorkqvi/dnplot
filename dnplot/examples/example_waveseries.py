import dnplot
import dnora as dn
point = dn.grid.Grid(lon=4.308, lat=62.838, name="Svinoy")
model = dn.modelrun.NORA3(point, year=2022, month=3, day=18)
model.import_spectra()
model.spectra_to_waveseries()
model.waveseries()

plot = dnplot.Dnora(model)
plot.waveseries(var=[('hs', 'tm01'),('hs', 'tm01'),('hs', 'tm01')])
