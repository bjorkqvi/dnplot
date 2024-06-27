import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from ..plot_functions import dnora_functions


class Dnora:
    def __init__(self, model):
        self.model = model

    def grid(self, grid_plotter=dnora_functions.grid_plotter) -> None:
        fig, ax = plt.subplots(1)
        fig_dict = {"fig": fig, "ax": ax}
        fig_dict = grid_plotter(fig_dict, self.model)
        fig_dict.get("fig").show()

    def wind(self, wind_plotter=dnora_functions.wind_plotter):
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
        fig_dict = wind_plotter(fig_dict, self.model)
        fig_dict.get("fig").show()

    def spectra(self, spectra_plotter=dnora_functions.spectra_plotter):
        fig, ax = plt.subplots(subplot_kw={"polar": True})
        fig_dict = {"fig": fig, "ax": ax}
        fig_dict = spectra_plotter(fig_dict, self.model)
        fig_dict.get("fig").show()
