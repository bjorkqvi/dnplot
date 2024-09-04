"""Dnplot takes a DNORA ModelRun-object, but can also be used with e.g. a dict of geo-skeleton Skeleton-objects

Here we will show how to use the grid plotter to plot topography data that is not read with DNORA."""

from geo_skeletons import GriddedSkeleton
from geo_skeletons.decorators import add_datavar
import dnplot
import numpy as np


@add_datavar("topo")
class Grid(GriddedSkeleton):
    pass


grid = Grid(lon=(5, 6), lat=(59, 60), name="DummyTopo")
grid.set_spacing(nx=100, ny=80)
grid.set_topo(np.random.rand(grid.ny(), grid.nx()) * 100)  # Dummy data

plot = dnplot.Matplotlib({"grid": grid})
plot.grid(block=True)
