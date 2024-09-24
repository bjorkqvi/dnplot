from dnplot import Matplotlib
from geo_skeletons import GriddedSkeleton, PointSkeleton
from geo_skeletons.decorators import add_datavar, add_mask


def test_plot_topo():
    @add_mask(
        name="sea",
        coord_group="grid",
        default_value=1,
        opposite_name="land",
        triggered_by="topo",
        valid_range=(0, None),
        range_inclusive=False,
    )
    @add_datavar("topo")
    class Grid(GriddedSkeleton):
        pass

    grid = Grid(lon=(5, 6), lat=(59, 61))
    grid.set_spacing(nx=10, ny=10)
    grid.set_topo(10)
    topo = grid.topo()
    topo[:, 0:3] = 0
    topo[5, :] = 5
    grid.set_topo(topo)
    plot = Matplotlib({"grid": grid})
    plot.topo()


def test_plot_grid():
    @add_mask(name="output", coord_group="grid", default_value=0)
    @add_mask(name="boundary", coord_group="grid", default_value=1)
    @add_datavar("topo")
    class Grid(GriddedSkeleton):
        pass

    grid = Grid(lon=(5, 6), lat=(59, 61))
    grid.set_spacing(nx=10, ny=10)
    grid.set_topo(10)
    mask = grid.boundary_mask() * False
    mask[5, 5] = True
    grid.set_output_mask(mask)

    wind = GriddedSkeleton(lon=(4.5, 6.5), lat=(58.5, 61.5))
    wind.set_spacing(dlon=0.25, dlat=0.25)
    current = GriddedSkeleton(lon=(4.4, 6.4), lat=(58.4, 61.4))
    current.set_spacing(dlon=0.125, dlat=0.125)
    spectra = PointSkeleton(lon=(5.1, 6.1, 5.5), lat=(60.0, 59.8, 60.2))
    waveseries = PointSkeleton(lon=(5.3, 6.3, 5.8), lat=(60.0, 59.8, 60.2))

    plot = Matplotlib(
        {
            "grid": grid,
            "wind": wind,
            "spectra": spectra,
            "waveseries": waveseries,
            "current": current,
        }
    )

    plot.grid()
