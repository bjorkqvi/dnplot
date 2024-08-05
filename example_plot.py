# Plotting with the new dnora version 2
#import dnora as dn
#import dnplot

# import os
# os.environ["DNORA_LOCAL_GRID_PATH"] = "~/bathy"

#grid = dn.grid.EMODNET(lon=(4, 6), lat=(59, 60))
#grid.import_topo(folder="~/kodai/bathy")
#grid.set_spacing(dm=5000)
#grid.mesh_grid()
#model = dn.modelrun.NORA3(grid, year=2022, month=2, day=1)
#model.import_wind()
#model.import_spectra()
#model.spectra_to_1d()

#plot = dnplot.Dnora(model)
#plot.spectra1d()


import dnora as dn
import dnplot

point=dn.grid.Grid(
    lon=(2.629880, 5.755490), lat=(59.517501, 60.748326), name="Bergen"
)

model = dn.modelrun.ModelRun(point, year=2017, month=2, day=5)
model.import_current(dn.current.read_metno.NorKyst800())
plot = dnplot.Dnora(model)
#breakpoint()
plot.current()
