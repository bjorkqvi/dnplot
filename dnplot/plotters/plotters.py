import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from ..plot_functions import matplotlib_functions
from ..plot_functions import plotly_functions
from typing import Callable


class Matplotlib:
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict

    def wavegrid(
        self,
        data_var: str,
        plotter: Callable = matplotlib_functions.wavegrid_plotter,
        coastline: bool = None,
        contour: bool = False,
    ):
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = None
        gl.right_labels = None
        fig_dict = {"fig": fig, "ax": ax, "gl": gl}
        fig_dict = plotter(
            fig_dict, self.data_dict, data_var, coastline=coastline, contour=contour
        )
        fig_dict.get("ax").legend()
        # if not test_mode:
        #     plt.show(block=True)

    def topo(
        self,
        plotter: Callable = matplotlib_functions.topo_plotter,
        coastline: bool = None,
        test_mode: bool = False,
        save_fig: bool = False,
    ) -> None:
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = None
        gl.right_labels = None
        fig_dict = {"fig": fig, "ax": ax, "gl": gl}
        fig_dict = plotter(fig_dict, self.data_dict, coastline=coastline)
        fig_dict.get("ax").legend()
        if not test_mode:
            if save_fig:
                fig_dict.get("fig").savefig(
                    "dnora_topo.png", bbox_inches="tight", dpi=300
                )
            else:
                plt.show(block=True)

    def grid(
        self,
        plotter: Callable = matplotlib_functions.grid_plotter,
        coastline: bool = None,
        test_mode: bool = False,
        save_fig: bool = False,
    ) -> None:
        fig, ax = plt.subplots(1)
        fig_dict = {"fig": fig, "ax": ax}
        fig_dict = plotter(fig_dict, self.data_dict, coastline=coastline)
        fig_dict.get("ax").legend()

        if not test_mode:
            if save_fig:
                fig_dict.get("fig").savefig(
                    "dnora_grid.png", bbox_inches="tight", dpi=300
                )
            else:
                plt.show(block=True)
            # fig_dict.get("fig").show()

    def wind(
        self,
        plotter: Callable = matplotlib_functions.directional_data_plotter,
        coastline: bool = True,
        contour: bool = True,
        test_mode: bool = False,
    ):
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = None
        gl.right_labels = None
        fig_dict = {"fig": fig, "ax": ax, "gl": gl}
        fig_dict = plotter(
            fig_dict,
            self.data_dict,
            obj_type="wind",
            coastline=coastline,
            contour=contour,
            test_mode=test_mode,
        )
        if not test_mode:
            plt.show(block=True)

    def current(
        self,
        plotter: Callable = matplotlib_functions.directional_data_plotter,
        coastline: bool = False,
        contour: bool = False,
        test_mode: bool = False,
    ):
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = None
        gl.right_labels = None
        fig_dict = {"fig": fig, "ax": ax, "gl": gl}
        fig_dict = plotter(
            fig_dict,
            self.data_dict,
            obj_type="current",
            coastline=coastline,
            contour=contour,
            test_mode=test_mode,
        )
        if not test_mode:
            plt.show(block=True)

    def spectra(
        self,
        plotter: Callable = matplotlib_functions.spectra_plotter,
        test_mode: bool = False,
    ):
        fig, ax = plt.subplots(subplot_kw={"polar": True})
        fig_dict = {"fig": fig, "ax": ax}
        fig_dict = plotter(fig_dict, self.data_dict)
        if not test_mode:
            plt.show(block=True)

    def waveseries(
        self,
        var=["hs", ("tm01", "tm02"), "dirm"],
        plotter: Callable = matplotlib_functions.waveseries_plotter,
        test_mode: bool = False,
    ):
        fig_dict = plotter(self.data_dict, var)

    def spectra1d(
        self,
        plotter: Callable = matplotlib_functions.spectra1d_plotter,
        test_mode: bool = False,
    ):
        fig, ax = plt.subplots()
        fig, ax2 = fig, ax.twinx()
        fig_dict = {"fig": fig, "ax": ax, "ax2": ax2}
        fig_dict = plotter(fig_dict, self.data_dict)
        if not test_mode:
            plt.show(block=True)


class Matplotlib1:
    def __init__(self, model, model1):
        self.data_dict = model
        self.data_dict1 = model1

    def scatter(
        self,
        var=["hs", "hs"],
        plotter: Callable = matplotlib_functions.scatter1_plotter,
    ):
        fig, ax = plt.subplots()
        fig_dict = {"fig": fig, "ax": ax}
        fig_dict = plotter(fig_dict, self.data_dict, self.data_dict1, var)


class Plotly:
    def __init__(self, model):
        self.data_dict = model

    def waveseries(
        self, use_dash, plotter: Callable = plotly_functions.waveseries_plotter
    ):
        fig_dict = plotter(self.data_dict, use_dash)

    def spectra(self, plotter: Callable = plotly_functions.spectra_plotter):
        fig_dict = plotter(self.data_dict)


class Plotly1:
    def __init__(self, model, model1):
        self.data_dict = model
        self.data_dict1 = model1

    def scatter(self, plotter: Callable = plotly_functions.scatter_plotter):
        fig_dict = plotter(self.data_dict, self.data_dict1)
