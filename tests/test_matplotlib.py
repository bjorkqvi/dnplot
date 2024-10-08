from dnplot import Matplotlib
from geo_skeletons import GriddedSkeleton, PointSkeleton
from geo_skeletons.decorators import add_datavar, add_mask, add_magnitude, add_time
import geo_parameters as gp


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


def test_plot_wind():

    @add_magnitude(gp.wind.Wind("mag"), x="u", y="v", direction=gp.wind.WindDir("dir"))
    @add_datavar(name=gp.wind.YWind("v"), default_value=0.0)
    @add_datavar(name=gp.wind.XWind("u"), default_value=0.0)
    @add_time(grid_coord=True)
    class Wind(GriddedSkeleton):
        pass

    wind = Wind(
        lon=(0, 10), lat=(50, 60), time=("2020-01-01 00:00", "2020-01-01 03:00")
    )
    wind.set_spacing(nx=11, ny=21)
    wind.set_u(5)
    wind.set_v(10)

    plot = Matplotlib(
        {
            "wind": wind,
        }
    )

    plot.wind(test_mode=True)


def test_plot_current():

    @add_magnitude(
        gp.ocean.Current("mag"), x="u", y="v", direction=gp.ocean.CurrentDir("dir")
    )
    @add_datavar(name=gp.ocean.YCurrent("v"), default_value=0.0)
    @add_datavar(name=gp.ocean.XCurrent("u"), default_value=0.0)
    @add_time(grid_coord=True)
    class Current(GriddedSkeleton):
        pass

    current = Current(
        lon=(0, 10), lat=(50, 60), time=("2020-01-01 00:00", "2020-01-01 03:00")
    )
    current.set_spacing(nx=11, ny=21)
    current.set_u(5)
    current.set_v(10)

    plot = Matplotlib(
        {
            "current": current,
        }
    )

    plot.current(test_mode=True)
