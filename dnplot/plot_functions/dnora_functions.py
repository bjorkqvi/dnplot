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
        ax=fig_dict['ax']
        ax2=fig_dict['ax2']
        ax.cla()
        ax2.cla()
        dirm=None
        spr=None
        if spectra1d.dirm() is not None:
            dirm=spectra1d.dirm()[sliders["time"].val, sliders["inds"].val, :]
        if spectra1d.spr() is not None:
            spr=spectra1d.spr()[sliders["time"].val, sliders["inds"].val, :]
            
        fig_dict = draw.draw_graph_spectra1d(
            fig_dict,
            spectra1d.spec()[sliders["time"].val, sliders["inds"].val, :],
            spectra1d.freq(),
            dirm,
            spr,
        )
        max_y = np.max(spectra1d.spec())
        upper_limit = ((max_y // 5 + 1) * 5)
        ax.set_ylim(0, upper_limit)
        ax.set_yticks(np.arange(0, upper_limit, 5))
        ax.set_title(spectra1d.name, fontsize=16)
        ax.set_xlabel('Frequency')
        ax.set_ylabel(f"{spectra1d.meta.get('spec').get('long_name')}\n {'E(f)'}", color='b')
        max_y1=np.max(spectra1d.dirm())
        upper_limit = ((max_y1 // 5 + 1) * 5)
        ax2.set_ylim(0, upper_limit)
        ax2.set_ylabel(f"{spectra1d.meta.get('dirm').get('long_name')}\n {spectra1d.meta.get('dirm').get('unit')}",color='g')
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        ax.grid()
        figure_initialized = True

    spectra1d = model.spectra1d()
    grid = model.grid()
    figure_initialized = False
    sliders = {}
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


from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


def waveseries_plotter(model:ModelRun):
    ts = model.waveseries()
    var = {
        'None': '',
        'UTC': ts.time(),
        'hs': ts.hs(),
        "tm01": ts.tm01(),
        "tm02": ts.tm02(),
        "tm_10": ts.tm_10(),
        "dirm": ts.dirm(),
    }

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1('Waveseries'),
        html.P("Select variable:"),
        dcc.Dropdown(
            id="waveseries-1",
            options=[
                {"label": "Hs", "value": "hs"},
                {"label": "Tm01", "value": "tm01"},
                {"label": "Tm02", "value": "tm02"},
                {"label": "Tm_10", "value": "tm_10"},
                {"label": "Dirm", "value": "dirm"},
            ],
            value="hs",
            clearable=False,
            style={'width': '30%'},
        ),
        dcc.Dropdown(
            id="waveseries-2",
            options=[
                {"label": "None", "value": "None"},
                {"label": "Hs", "value": "hs"},
                {"label": "Tm01", "value": "tm01"},
                {"label": "Tm02", "value": "tm02"},
                {"label": "Tm_10", "value": "tm_10"},
                {"label": "Dirm", "value": "dirm"},
            ],
            value="None",
            clearable=False,
            style={'width': '30%'},
        ),
        dcc.Graph(id="waveseries_chart"),
    ])

    @app.callback(
        Output("waveseries_chart", "figure"), 
        Input("waveseries-1", "value"), 
        Input("waveseries-2", "value"),
    )
    def display_time_series(ticker1, ticker2):
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        fig = px.line(var, x='UTC', y=ticker1)
        subfig.add_trace(fig.data[0], secondary_y=False)
        if ticker2 != "None":
            fig2 = px.line(var, x='UTC', y=ticker2)
            subfig.add_trace(fig2.data[0], secondary_y=True)
            subfig.update_traces(line_color='blue', secondary_y=False)
            subfig.update_traces(line_color='red', secondary_y=True)
            subfig.update_xaxes(minor=dict(ticks="inside", showgrid=True))
            subfig.update_yaxes(secondary_y=True, showgrid=False)
            subfig.update_layout(xaxis_title="UTC", yaxis_title=ticker1)
            subfig.update_yaxes(title_text=ticker2, secondary_y=True)
        else:
            subfig.update_layout(xaxis_title="UTC", yaxis_title=ticker1)
        
        return subfig
    #default is #app.run_server()debug=True)
    #http://127.0.0.1:8050/, if it says is in use, than change it as you see
    #app.run_server("0.0.0.0", 8000,debug=True)
    app.run_server("0.0.0.0", 8000, debug=True)