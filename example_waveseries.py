import dnplot
import dnora as dn
point = dn.grid.Grid(lon=4.308, lat=62.838, name="Svinoy")
model = dn.modelrun.NORA3(point, year=2022, month=3, day=18)
model.import_spectra()
model.spectra_to_waveseries()
model.waveseries()

e39 = dn.modelrun.ModelRun(year=2019, month=3)
e39.import_waveseries(dn.waveseries.read.E39(loc="D"), point_picker=dn.pick.Trivial())

point = dn.grid.Grid(lon=e39.waveseries().lon(), lat=e39.waveseries().lat())
nora3 = dn.modelrun.NORA3(point, year=2019, month=3)
nora3.import_spectra()
nora3.spectra_to_waveseries()



#plot = dnplot.Dnora(model)

plot = dnplot.Plotly1(e39, nora3)
#plot.waveseries()
plot.scatter()
