import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde
import os

from dash import Input, Output
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from threading import Timer
import webbrowser
import random
from flask import Flask
import cmocean.cm

from dnplot.stats import calculate_correlation, calculate_RMSE
from dnplot import sanitation
from dnplot.plot_functions import plotly_layout
from dnplot.draw_functions import plotly_draw
from dash import Dash, dcc, html
def open_browser(port):
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new(f"http://127.0.0.1:{port}/")


def waveseries_plotter_basic(model, model1):
    xmodel = sanitation.force_to_ds(model)
    xdf = sanitation.xarray_to_dataframe(xmodel)
    ymodel = sanitation.force_to_ds(model1)
    if ymodel is not None:
        ydf = sanitation.xarray_to_dataframe(ymodel)

    if ymodel is not None:
        df = pd.merge(
            xdf.set_index("time").add_suffix(f" {xmodel.name}").reset_index(),
            ydf.set_index("time").add_suffix(f" {ymodel.name}").reset_index(),
            on="time",
        )
    else:
        df = xdf


    fig = go.Figure()

    variables = [col for col in df if col != 'time']
    for variable in variables:
        trace = go.Scatter(
            x=df["time"],
            y=df[variable],
            mode="lines",
            name=variable,
            visible="legendonly",
        )
        fig.add_trace(trace)

    fig.update_layout(title=f"{xmodel.name}", xaxis_title="UTC", yaxis_title="Values")
    fig.show()


def waveseries_plotter_dash(model, model1):
    xmodel_all = sanitation.force_to_ds(model)
    xlon, xlat = xmodel_all.lon.values, xmodel_all.lat.values
    ymodel_all = sanitation.force_to_ds(model1)
    xname = xmodel_all.name
    if ymodel_all is not None:
        ylon, ylat = ymodel_all.lon.values, ymodel_all.lat.values
        yname = ymodel_all.name
    else:
        ylon, ylat = None, None
        yname = None
    app = plotly_layout.create_wave_data_layout(xmodel_all, ymodel_all)

    @app.callback(
        Output("data_chart", "figure"),
        Output("title", "children"),
        Output("map", "figure"),
        Input("dropdown1", "value"),
        Input("dropdown2", "value"),
        Input("xslider", "value"),
        Input("yslider", "value"),
        Input('map','relayoutData'),
    )
    def display_time_series(var1, var2, inds_x, inds_y, relayout_data):
        __, __, df = sanitation.get_one_point_merged_dataframe(xmodel_all, ymodel_all, inds_x, inds_y)
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        fig = px.line(df, x="time", y=var1)
        subfig.add_trace(fig.data[0], secondary_y=False)
        if var2 != "None":
            fig2 = px.line(df, x="time", y=var2)
            subfig.add_trace(fig2.data[0], secondary_y=True)
            subfig.update_traces(line_color="blue", secondary_y=False)
            subfig.update_traces(line_color="red", secondary_y=True)
            subfig.update_xaxes(minor=dict(ticks="inside", showgrid=True))
            subfig.update_yaxes(secondary_y=True, showgrid=False)
            
            subfig.update_yaxes(title_text=var2, secondary_y=True)

        if ymodel_all is None:
            subfig.update_layout(xaxis_title="UTC", yaxis_title=var1, title=f'{xmodel_all.name} (lat: {xlat[inds_x]:.3f}, lon: {xlon[inds_x]:.3f})')
        else:
            subfig.update_layout(xaxis_title="UTC", yaxis_title=var1, title=f'{xmodel_all.name} (lat: {xlat[inds_x]:.3f}, lon: {xlon[inds_x]:.3f}); {ymodel_all.name} (lat: {ylat[inds_y]:.3f}, lon: {ylon[inds_y]:.3f})')


        subfig.update_layout(
            margin=dict(l=0, r=0, t=50, b=50),
        )
        fig = plotly_draw.draw_map(xlon, xlat, ylon, ylat, inds_x, inds_y, relayout_data, xname, yname)

        if ymodel_all is not None:
            title = f"{xmodel_all.name} and {ymodel_all.name} Waveseries"
        else:
            title = f"{xmodel_all.name} Waveseries"
        return subfig, title, fig

    port = random.randint(1000, 9999)
    Timer(1, open_browser, args=[port]).start()
    app.run(debug=False, port=port)


def waveseries_plotter(model, model1, plain: bool):
    if plain:
        waveseries_plotter_basic(model, model1)
    else:
        waveseries_plotter_dash(model, model1)        



def scatter_plotter(model, model1):
    xmodel_all = sanitation.force_to_ds(model)
    xlon, xlat = xmodel_all.lon.values, xmodel_all.lat.values
    xname = xmodel_all.name
    ymodel_all = sanitation.force_to_ds(model1)
    if ymodel_all is None:
        ylon, ylat = None, None
        yname = None
    else:
        ylon, ylat = ymodel_all.lon.values, ymodel_all.lat.values
        yname = ymodel_all.name
    app = plotly_layout.create_wave_data_layout(xmodel_all, ymodel_all, set_y_start_val=True)

    @app.callback(
        Output("data_chart", "figure"),
        Output("title", "children"),
        Output("map", "figure"),
        Input("dropdown1", "value"),
        Input("dropdown2", "value"),
        Input("xslider", "value"),
        Input("yslider", "value"),
        Input('map','relayoutData'),
    )
    def update_graph(xvar, yvar, inds_x, inds_y, relayout_data):
        __, __, df = sanitation.get_one_point_merged_dataframe(xmodel_all, ymodel_all, inds_x, inds_y)

        xdata, ydata = df[xvar].values, df[yvar].values
        RMSE = np.sqrt(np.mean((xdata-ydata)**2))
        R = np.corrcoef(xdata, ydata)[0,1]
        SI = RMSE / np.mean(xdata)*100
        xy = np.vstack([xdata, ydata])
        z = gaussian_kde(xy)(xy)

        if xvar not in df.columns or yvar not in df.columns:
            return go.Figure()
        
        # Add scatter
        xunit = sanitation.get_units(xmodel_all,xvar.split(' ')[0])
        if ymodel_all is not None:
            yunit = sanitation.get_units(ymodel_all,yvar.split(' ')[0])
        else:
            yunit = sanitation.get_units(xmodel_all,yvar.split(' ')[0])
        fig = px.scatter(
           df, x=xvar, y=yvar, color=z, color_continuous_scale='blues', labels={
                xvar: f"{xvar} ({xunit})",
                yvar: f"{yvar} ({yunit})"}
        )

        # Lines
        x_values = np.linspace(0, np.ceil(np.max(xdata)), 100)
        
        slope, intercept = np.polyfit(xdata, ydata,1)
        fig.add_traces(
            go.Scatter(
                x=x_values,
                y=x_values*slope+intercept,
                mode="lines",
                name="Linear regression",
                visible=True,
            )
        )

       
        fig.add_traces(
            go.Scatter(
                x=x_values, y=x_values, mode="lines", name="x=y", visible="legendonly"
            )
        )
        
        a = np.mean(ydata)/np.mean(xdata)
        fig.add_traces(
            go.Scatter(
                x=x_values,
                y=a * x_values,
                mode="lines",
                name="one-parameter-linear regression",
                visible="legendonly",
            )
        )

        maxval = np.maximum(np.max(xdata), np.max(ydata))
        fig.update_layout(
            yaxis=dict(range=[0, maxval]), xaxis=dict(range=[0, maxval])
        )


        xvarname = sanitation.get_varname(xmodel_all,xvar.split(' ')[0])
        if ymodel_all is not None:
            yvarname = sanitation.get_varname(ymodel_all,yvar.split(' ')[0])
        else:
            yvarname = sanitation.get_varname(xmodel_all,yvar.split(' ')[0])
        
        text = [f"N={len(xdata)}"]
        if xunit == yunit:
            text.append(f"Bias={np.mean(xdata)-np.mean(ydata):.2f}{xunit}")
            text.append(f"RMSE={RMSE:.2f}{xunit}")
            text.append(f"SI={SI:.0f}%")
        text.append(f"r={R:.2f}")
        text = '; '.join(text)

        fig.update_layout(
            coloraxis_colorbar=dict(title="Density", y=0.45, x=1.015, len=0.9),
            annotations=[
                dict(
                    x=0.001,
                    y=0.995,
                    xref="paper",
                    yref="paper",
                    text=text,
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    align="left",
                    bgcolor="white",
                    borderpad=4,
                    bordercolor="black",
                    opacity=0.55,
                )
            ],
        )


        fig.update_layout(
            width=800, 
            height=800, 
            margin=dict(l=0, r=0, t=40, b=0)
        )

        mapfig = plotly_draw.draw_map(xlon, xlat, ylon, ylat, inds_x, inds_y, relayout_data, xname, yname)
        
        if yname is not None:
            title = f"{xname} and {yname} scatter"
        else:
            title = f"{xname} scatter"
        
        return fig, title, mapfig

    port = random.randint(1000, 9999)
    Timer(1, open_browser, args=[port]).start()
    app.run(debug=False, port=port)



def spectra_plotter(model, model1):
    spectra = model.spectra()
    spectra1d = model.spectra1d()

    if model1:
        spectra_B = model1.spectra()
        spectra1d_B = model1.spectra1d()
    else:
        spectra_B, spectra1d_B = None, None

    if spectra is not None and spectra_B is not None:
        spectra, spectra_B = spectra.cut_to_common_times(spectra_B)

    if spectra1d is not None and spectra1d_B is not None:
        spectra1d, spectra1d_B = spectra1d.cut_to_common_times(spectra1d_B)

    number_of_plots = 0
    if spectra1d is not None:
        number_of_plots += 1
        primary_object = "spectra1d"

    if spectra1d_B is not None:
        primary_object_B = "spectra1d"

    if spectra is not None or spectra_B is not None:
        number_of_plots += 1
        primary_object = "spectra"

    if spectra_B is not None: 
        primary_object_B = "spectra"

    lons, lats = model[primary_object].lonlat()
    times = model[primary_object].time(datetime=False)
    name = model[primary_object].name

    if model1:
        lons_B, lats_B = model1[primary_object_B].lonlat()
        name_B = model1[primary_object_B].name
    else:
        lons_B, lats_B = None, None
        name_B = None


    app = Dash(__name__)
    
    app.layout = plotly_layout.create_spectra_app_layout(
        len_of_inds=len(lons),
        len_of_inds_B=len(lons_B) if lons_B is not None else 0,
        len_of_times=len(times),
        number_of_plots=number_of_plots,
        name=name, 
        name_B=name_B
    )

    outputs = [
        Output("title", "children"),
        Output("smaller_title", "children"),
        Output("spectra_map", "figure"),
        Output("primary_graph", "figure"),
    ]
    if number_of_plots == 2:
        outputs.append(Output("secondary_graph", "figure"))

    @app.callback(
        outputs,
        [Input("time_slider", "value"), Input("inds_slider", "value"), Input("inds_slider_B", "value"),Input('spectra_map','relayoutData')],
    )
    def display_spectra(time_r, inds_r, inds_B, relayout_data):
        spectra_map = plotly_draw.draw_map(lons, lats, lons_B, lats_B, inds_r, inds_B, relayout_data, name, name_B)
        spectra_map.update_layout(
            margin=dict(l=50, r=50, t=0, b=0)
        )

        graphs = {}
        if spectra is not None:
            spec1 = spectra.spec(squeeze=False)[:, inds_r, :, :].flatten()
            graphs["spectra"] = plotly_draw.draw_plotly_graph_spectra(
                freq=spectra.freq(),
                spec=spectra.spec(squeeze=False)[time_r, inds_r, :, :].flatten(),
                dirs=spectra.dirs(),
                cmin=np.min(spec1),
                cmax=np.max(spec1),
            )

            time = spectra1d.time(datetime=False)[time_r]
            lon, lat = spectra1d.lon()[inds_r], spectra1d.lat()[inds_r]
            graphs["spectra"].update_layout(
                #width=800,
                #height=800,
                margin=dict(l=0, r=0, t=50, b=0),
                title=dict(
                text=f"{spectra.name}\n{time}; lat={lat:.4f}, lon={lon:.4f}",  # The title text
                font=dict(size=18),  # Font size for the title
                x=0.5,  # Center the title horizontally (0 = left, 1 = right)
                y=1,  # Position the title near the top of the plot
                xanchor="center",  # Anchor the title horizontally by its center
                yanchor="top",  # Anchor the title vertically by its top
                ),
            )


        maxdir, mindir = None, None
        maxdir_B, mindir_B = None, None
        if spectra1d is not None:
            spec1d = spectra1d.spec(squeeze=False)[:, inds_r, :].flatten()
            dirm = spectra1d.dirm(squeeze=False)[time_r, inds_r, :] if spectra1d.dirm() is not None else None
            spr = spectra1d.spr(squeeze=False)[time_r, inds_r, :] if spectra1d.spr() is not None else None
            graphs["spectra1d"] = plotly_draw.draw_plotly_graph_spectra1d(
                freq=spectra1d.freq(),
                spec=spectra1d.spec(squeeze=False)[time_r, inds_r, :],
                dirm=dirm,
                spr=spr,
                name=name,
            )
            maxdir = dirm
            mindir = dirm
            if spr is not None and maxdir is not None:
                maxdir = maxdir + spr
                mindir = mindir - spr

        if spectra1d_B is not None:
            spec1d = spectra1d_B.spec(squeeze=False)[:, inds_B, :].flatten()
            dirm_B = spectra1d_B.dirm(squeeze=False)[time_r, inds_r, :] if spectra1d.dirm() is not None else None
            spr_B = spectra1d_B.spr(squeeze=False)[time_r, inds_r, :] if spectra1d.spr() is not None else None
            
            graphs["spectra1d"] = plotly_draw.draw_plotly_graph_spectra1d(
                freq=spectra1d_B.freq(),
                spec=spectra1d_B.spec(squeeze=False)[time_r, inds_B, :],
                dirm=dirm_B,
                spr=spr_B,
                name=name_B,
                fig=graphs["spectra1d"]
            )
            maxdir_B = dirm_B
            mindir_B = dirm_B
            if spr_B is not None and maxdir_B is not None:
                maxdir_B = maxdir_B + spr_B
                mindir_B = mindir_B - spr_B



        if spectra1d is not None:
            if maxdir_B is not None:
                maxdir = np.ceil(np.maximum(np.nanmax(maxdir), np.nanmax(maxdir_B))/10)*10+10
            else:
                maxdir = np.ceil(np.nanmax(maxdir)/10)*10+10
            
            if mindir_B is not None:
                mindir = np.floor(np.minimum(np.nanmin(mindir), np.nanmin(mindir_B))/10)*10-10
            else:
                mindir = np.floor(np.nanmin(mindir)/10)*10-10

            time = spectra1d.time(datetime=False)[time_r]
            lon, lat = spectra1d.lon()[inds_r], spectra1d.lat()[inds_r]

            graphs["spectra1d"].update_layout(
                title=dict(
                text=f"{time}; lat={lat:.4f}, lon={lon:.4f}",  # The title text
                font=dict(size=18),  # Font size for the title
                x=0.5,  # Center the title horizontally (0 = left, 1 = right)
                y=1,  # Position the title near the top of the plot
                xanchor="center",  # Anchor the title horizontally by its center
                yanchor="top",  # Anchor the title vertically by its top
                ),
                xaxis_title=f"{spectra1d.meta.get('freq').get('long_name')}\n (Hz)",
                yaxis=dict(
                    title=f"{spectra1d.meta.get('spec').get('long_name')}\n E(f) ({spectra1d.meta.get('spec').get('units')})",
                    range=[0, np.max(spec1d) * 1.1],
                ),
                yaxis2=dict(
                    title=f"{spectra1d.meta.get('dirm').get('long_name')}\n ({spectra1d.meta.get('dirm').get('units')})",
                    overlaying="y",
                    side="right",
                    range=[mindir, maxdir],
                ),
                #width=800,
                #height=500,
                margin=dict(l=0, r=0, t=50, b=0),
            )

        title = f"{times[time_r]} {name}"
        if name_B is not None:
            title = f"{title} and {name_B}"
        smaller_title = f"Latitude={lats[inds_r]:.4f} Longitude={lons[inds_r]:.4f}"

        if number_of_plots == 1:
            return title, smaller_title, spectra_map, graphs.get(primary_object)
        else:
            return (
                title,
                smaller_title,
                spectra_map,
                graphs.get("spectra"),
                graphs.get("spectra1d"),
            )

    port = random.randint(1000, 9999)
    Timer(1, open_browser, args=[port]).start()
    app.run(debug=False, port=port)


