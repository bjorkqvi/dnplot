import dnora as dn
import dnplot

grid = dn.grid.Grid(lon=(4, 6), lat=(59, 60), name="Haugesund")
model = dn.modelrun.NORA3(grid, year=2022, month=2, day=1)
model.import_spectra()
model.spectra_to_1d()

plot = dnplot.Matplotlib(model)
plot.spectra1d()
