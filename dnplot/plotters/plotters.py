import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from ..plot_functions import matplotlib_functions
from ..plot_functions import plotly_functions
from typing import Callable


class Matplotlib:
    def __init__(self, data_dict: dict, data_dict2: dict = None):
        self.data_dict = data_dict
        self.data_dict2 = data_dict2 or {}

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
        lon:float = None, 
        lat:float = None, 
        separate_plots: bool = None, 
        test_mode: bool = False,
    ):
        """var = ['hs', 'tp'] or ['hs',('tp','tm01')]

        use 'separate_plots = True' to get separate figures for each parameter.
        Default is separate_plots = True for more than 4 parameters.
        
        Use 'lon', 'lat' to pick a point to plot in case the data has many."""
        fig_dict = plotter(self.data_dict, self.data_dict2, var, lon, lat, separate_plots)

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


    def scatter(
        self,
        xvar="hs",
        yvar='hs',
        plotter: Callable = matplotlib_functions.scatter_plotter,
    ):
        fig, ax = plt.subplots()
        fig_dict = {"fig": fig, "ax": ax}
        data_dict2 = self.data_dict2 or self.data_dict
        fig_dict = plotter(fig_dict, self.data_dict, data_dict2, xvar, yvar)




class Plotly:
    def __init__(self, data_dict: dict, data_dict2: dict = None):
        self.data_dict = data_dict
        self.data_dict2 = data_dict2 or {}
    def waveseries(
        self, plain: bool=False, plotter: Callable = plotly_functions.waveseries_plotter
    ):
        fig_dict = plotter(self.data_dict, self.data_dict2, plain)

    def spectra(self, plotter: Callable = plotly_functions.spectra_plotter):
        fig_dict = plotter(self.data_dict, self.data_dict2)

    def scatter(self, plotter: Callable = plotly_functions.scatter_plotter):
        fig_dict = plotter(self.data_dict, self.data_dict2)

