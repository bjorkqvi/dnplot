import dnora as dn
import dnplot

grid = dn.grid.EMODNET(lon=(4, 6), lat=(59, 60), name="Haugesund")
grid.import_topo(folder="~/Documents/bathy")
grid.set_spacing(dm=500)
grid.mesh_grid()
model = dn.modelrun.NORA3(grid, year=2022, month=2, day=1)
model.import_wind()

plot = dnplot.Matplotlib(model)
plot.grid(block=True)
