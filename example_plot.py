# Plotting with the new dnora version 2
import dnora as dn
import dnplot

# import os
# os.environ["DNORA_LOCAL_GRID_PATH"] = "~/bathy"

grid = dn.grid.EMODNET(lon=(4, 6), lat=(59, 60))
grid.import_topo(folder="~/bathy")
grid.set_spacing(dm=500)
grid.mesh_grid()
model = dn.modelrun.NORA3(grid, year=2020, month=1, day=1)
model.import_wind()
model.import_spectra()

plot = dnplot.Dnora(model)
plot.grid()
breakpoint()
