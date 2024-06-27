from ..draw_functions import draw
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from ..defaults import default_variable
from dnora.dnora_type_manager.dnora_types import DnoraDataType


def grid_plotter(fig_dict, model) -> dict:
    """Plot the depth information and set output points etc."""
    grid = model.grid()
    fig_dict = draw.draw_gridded_magnitude(
        fig_dict,
        grid.x(native=True),
        grid.y(native=True),
        grid.topo(),
        cmap=default_variable["topo"]["cmap"],
    )
    fig_dict = draw.draw_mask(
        fig_dict, grid.x(native=True), grid.y(native=True), grid.land_mask()
    )

    fig_dict["ax"].set_xlabel(grid.core.x_str)
    fig_dict["ax"].set_ylabel(grid.core.y_str)
    fig_dict["cbar"].set_label("Depth [m]")
    fig_dict["ax"].set_title(f"{grid.name} {grid.ds().attrs.get('source', '')}")

    # masks_to_plot = ["spectra_mask", "output_mask"]
    # fig_dict = draw.draw_masked_points(fig_dict, grid, masks_to_plot=masks_to_plot)

    objects_to_plot = [
        DnoraDataType.WIND,
    ]
    fig_dict = draw.draw_object_points(fig_dict, model, objects_to_plot=objects_to_plot)

    fig_dict.get("ax").legend()

    return fig_dict


def wind_plotter(fig_dict, model) -> dict:
    def update_plot(val):
        nonlocal fig_dict
        nonlocal figure_initialized
        fig_dict = draw.draw_gridded_magnitude(
            fig_dict,
            wind.x(native=True),
            wind.y(native=True),
            wind.magnitude()[val, :, :],
            vmax=np.max(wind.magnitude()),
            vmin=0,
            cmap=default_variable["ff"]["cmap"],
        )
        fig_dict = draw.draw_coastline(fig_dict)
        fig_dict = draw.draw_arrows(
            fig_dict,
            wind.x(native=True),
            wind.y(native=True),
            wind.u()[val, :, :],
            wind.v()[val, :, :],
        )
        # if not figure_initialized:
        #     masks_to_plot = ["output_mask"]
        #     fig_dict = draw.draw_masked_points(fig_dict, grid, masks_to_plot=masks_to_plot)
        #     fig_dict.get("ax").legend()
        fig_dict["ax"].set_title(f"{wind.time(datetime=False)[val]} {wind.name}")
        figure_initialized = True

    wind = model.wind()
    grid = model.grid()
    figure_initialized = False
    if len(wind.time()) > 1:
        ax_slider = plt.axes([0.17, 0.05, 0.65, 0.03])
        time_slider = Slider(
            ax_slider, "time_index", 0, len(wind.time()) - 1, valinit=0, valstep=1
        )
        time_slider.on_changed(update_plot)

    update_plot(0)
    fig_dict["ax"].set_xlabel(wind.core.x_str)
    fig_dict["ax"].set_ylabel(wind.core.y_str)
    fig_dict["cbar"].set_label("Wind speed [m/s]")

    plt.show(block=True)

    return fig_dict


def spectra_plotter(fig_dict, model) -> dict:
    def update_plot(val):
        nonlocal fig_dict
        nonlocal figure_initialized
        fig_dict = draw.draw_polar_spectra(
            fig_dict,
            spectra.spec()[sliders["time"].val, sliders["inds"].val, :, :],
            spectra.freq(),
            spectra.dirs(),
        )

        fig_dict["ax"].set_title(
            f"{wind.time(datetime=False)[sliders['time'].val]} {wind.name}"
        )
        figure_initialized = True

    wind = model.wind()
    spectra = model.spectra()
    grid = model.grid()
    figure_initialized = False
    sliders = {}
    if len(wind.time()) > 1:
        ax_slider = plt.axes([0.17, 0.05, 0.65, 0.03])
        sliders["time"] = Slider(
            ax_slider, "time_index", 0, len(wind.time()) - 1, valinit=0, valstep=1
        )
        sliders["time"].on_changed(update_plot)
    if len(spectra.inds()) > 1:
        ax_slider2 = plt.axes([0.17, 0.01, 0.65, 0.03])
        sliders["inds"] = Slider(
            ax_slider2, "inds_index", 0, len(spectra.x()) - 1, valinit=0, valstep=1
        )
        sliders["inds"].on_changed(update_plot)
    update_plot(0)
    # fig_dict['ax'].set_xlabel(wind.core.x_str)
    # fig_dict['ax'].set_ylabel(wind.core.y_str)
    # fig_dict['cbar'].set_label('Wind speed [m/s]')

    plt.show(block=True)

    return fig_dict
