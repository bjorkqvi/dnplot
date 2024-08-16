# Plotting with the new dnora version 2
import dnora as dn
import dnplot

# import os
# os.environ["DNORA_LOCAL_GRID_PATH"] = "~/bathy"

grid = dn.grid.EMODNET(lon=(4, 6), lat=(59, 60))
grid.import_topo(folder="~/kodai/bathy")
grid.set_spacing(dm=5000)
grid.mesh_grid()
model = dn.modelrun.NORA3(grid, year=2022, month=2, day=1)
model.import_wind()
model.import_spectra()
model.spectra_to_1d()

plot = dnplot.Dnora(model)
plot.spectra1d()
