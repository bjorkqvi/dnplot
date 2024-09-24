from dnplot import Matplotlib
from geo_skeletons import GriddedSkeleton
from geo_skeletons.decorators import add_datavar


def test_plot_grid():
    @add_datavar("topo")
    class Grid(GriddedSkeleton):
        pass

    grid = Grid(lon=(5, 6), lat=(59, 61))
    grid.set_spacing(nx=10, ny=10)
    grid.set_topo(10)

    plot = Matplotlib({"grid": grid})
    plot.grid()
