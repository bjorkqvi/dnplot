from __future__ import annotations
from ..draw_functions import draw
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from ..defaults import default_variable
from dnora.dnora_type_manager.dnora_types import DnoraDataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dnora.modelrun import ModelRun


def grid_plotter(fig_dict: dict, model: ModelRun) -> dict:
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


def wind_plotter(fig_dict: dict, model: ModelRun) -> dict:
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


def spectra_plotter(fig_dict: dict, model: ModelRun) -> dict:
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
            f"{spectra.time(datetime=False)[sliders['time'].val]} {spectra.name}"
        )
        figure_initialized = True

    spectra = model.spectra()
    grid = model.grid()
    figure_initialized = False
    sliders = {}
    if len(spectra.time()) > 1:
        ax_slider = plt.axes([0.17, 0.05, 0.65, 0.03])
        sliders["time"] = Slider(
            ax_slider, "time_index", 0, len(spectra.time()) - 1, valinit=0, valstep=1
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

def waveseries_plotter(fig_dict: dict, model: ModelRun):
    var = fig_dict['var']
    ts = model.waveseries()
    if len(var) < 4:
        fig, axes = plt.subplots(len(var), 1)
        fig.suptitle(ts.name,fontsize=16)
        axes = axes if len(var) > 1 else [axes] 
        for i, item in enumerate(var):
            if isinstance(item, tuple):
                var1, var2 = item
                ax=axes[i]
                ax.plot(ts.get('time'),ts.get(var1),color='b',label=f"{var1} ({ts.meta.get(var1)['unit']})")
                
                ax.set_ylabel(f"{ts.meta.get(var1)['long_name']}\n ({ts.meta.get(var1)['unit']})",color='b')
                ax.set_xlabel('UTC',fontsize=12)
                ax2=ax.twinx()
                ax2.plot(ts.get('time'),ts.get(var2),color='g',label=f"{var2} ({ts.meta.get(var2)['unit']})")
                ax2.set_ylabel(f"{ts.meta.get(var2)['long_name']}\n ({ts.meta.get(var2)['unit']})",color='g')
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
                ax.grid(True)
            
            else:
                axes[i].plot(ts.get('time'),ts.get(item),color='b',label=f"{item} ({ts.meta.get(item)['unit']})")
                axes[i].set_ylabel(f"{ts.meta.get(item)['long_name']} \n ({ts.meta.get(item)['unit']})")
                axes[i].set_xlabel('UTC',fontsize=12)
                axes[i].legend()
                axes[i].grid(True)
        plt.tight_layout()
        plt.show()

    else:
        for item in var:
            fig, ax = plt.subplots()
            if isinstance(item, tuple):
                var1, var2 = item
                ax.plot(ts.get('time'),ts.get(var1),color='b',label=f"{var1} ({ts.meta.get(var1)['unit']})")

                ax.set_ylabel(f"{ts.meta.get(var1)['long_name']}\n ({ts.meta.get(var1)['unit']})",color='b')
                ax.set_xlabel('UTC',fontsize=12)
                ax2=ax.twinx()
                ax2.plot(ts.get('time'),ts.get(var2),color='g',label=f"{var2} ({ts.meta.get(var2)['unit']})")
                ax2.set_ylabel(f"{ts.meta.get(var2)['long_name']}\n ({ts.meta.get(var2)['unit']})",color='g')
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
                ax.grid(True)
            else:
                ax.plot(ts.get('time'),ts.get(item),color='b',label=f"{item} ({ts.meta.get(item)['unit']})")
                ax.set_xlabel('UTC',fontsize=12)
                ax.set_ylabel(f"{ts.meta.get(item)['long_name']} \n ({ts.meta.get(item)['unit']})")
                ax.legend()
                ax.grid(True)
            ax.set_title(ts.name,fontsize=16)
        
        plt.tight_layout()
        plt.show()


def spectra1d_plotter(fig_dict: dict, model: ModelRun) -> dict:
    def update_plot(val):
        nonlocal fig_dict
        nonlocal figure_initialized
        fig_dict['ax'].cla()
        fig_dict = draw.draw_graph_spectra1d(
            fig_dict,
            spectra1d.spec()[sliders["time"].val, sliders["inds"].val, :],
            spectra1d.freq(),
        )
        figure_initialized = True
        max_y=np.max(spectra1d.spec())
        upper_limit=((max_y/5+1)*5)
        fig_dict['ax'].set_ylim(0,upper_limit)
        fig_dict['ax'].set_yticks(np.arange(0, upper_limit, 5))
        fig_dict['ax'].set_ylim(0,upper_limit)
        fig_dict['ax'].set_title(spectra1d.name, fontsize=16)
        fig_dict['ax'].set_ylabel(f"{'Wave spectrum'}\n {'E(f)'}")
        fig_dict['ax'].set_xlabel('Frequency')                    
        fig_dict['ax'].grid()

    spectra1d = model.spectra1d()
    grid = model.grid()
    figure_initialized = False
    sliders={}
    if len(spectra1d.time()) > 1:
        ax_slider = plt.axes([0.17, 0.05, 0.65, 0.03])
        sliders["time"] = Slider(
            ax_slider, "time_index", 0, len(spectra1d.time()) - 1, valinit=0, valstep=1
        )
        sliders["time"].on_changed(update_plot)
    if len(spectra1d.inds()) > 1:
        ax_slider2 = plt.axes([0.17, 0.01, 0.65, 0.03])
        sliders["inds"] = Slider(
            ax_slider2, "inds_index", 0, len(spectra1d.x()) - 1, valinit=0, valstep=1
        )
        sliders["inds"].on_changed(update_plot)
    update_plot(0)
    plt.show(block=True)
    return fig_dict