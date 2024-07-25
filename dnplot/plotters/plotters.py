from __future__ import annotations
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from ..plot_functions import dnora_functions
from ..plot_functions import plotly_functions
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from dnora.modelrun import ModelRun


class Dnora:
    def __init__(self, model: ModelRun):
        self.model = model

    def grid(self, plotter: Callable = dnora_functions.grid_plotter) -> None:
        fig, ax = plt.subplots(1)
        fig_dict = {"fig": fig, "ax": ax}
        fig_dict = plotter(fig_dict, self.model)
        fig_dict.get("fig").show()

    def wind(self, plotter: Callable = dnora_functions.wind_plotter):
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
        fig_dict = plotter(fig_dict, self.model)
        fig_dict.get("fig").show()

    def spectra(self, plotter: Callable = dnora_functions.spectra_plotter):
        fig, ax = plt.subplots(subplot_kw={"polar": True})
        fig_dict = {"fig": fig, "ax": ax}
        fig_dict = plotter(fig_dict, self.model)
        fig_dict.get("fig").show()

    def waveseries(self, var, plotter: Callable = dnora_functions.waveseries_plotter):
        #fig, ax = plt.subplots(1)
        fig_dict = { "var": var}
        fig_dict = plotter(fig_dict, self.model)
        #fig_dict.get("fig").show()  

    def spectra1d(self, plotter: Callable = dnora_functions.spectra1d_plotter):
        fig, ax = plt.subplots()
        fig, ax2= fig, ax.twinx()
        fig_dict = {"fig": fig, "ax": ax, "ax2":ax2}
        fig_dict = plotter(fig_dict, self.model)
        fig_dict.get("fig").show()

class Plotly:
    def __init__(self, model: ModelRun):
        self.model = model

    def waveseries(self, plotter: Callable =plotly_functions.waveseries_plotter):
        fig_dict = plotter(self.model)
