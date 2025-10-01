from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import os
from threading import Timer
import webbrowser
import random
from flask import Flask
from dnplot.stats import calculate_correlation, calculate_RMSE
from dnplot import sanitation
import cmocean.cm
def xarray_to_dataframe(model) -> pd.DataFrame:
    df = model.ds().to_dataframe()
    df = df.reset_index()
    col_drop = ["lon", "lat", "inds"]
    df = df.drop(col_drop, axis="columns")
    df.set_index("time", inplace=True)
    df = df.resample("h").asfreq()
    df = df.reset_index()
    return df




def linear_regression_line(xdata, ydata, fig):
    slope, intercept = np.polyfit(xdata, ydata,1)
    x_range = np.linspace(0, np.ceil(np.max(xdata)), 100)
    fig.add_traces(
        go.Scatter(
            x=x_range,
            y=x_range*slope+intercept,
            mode="lines",
            name="Linear regression",
            visible=True,
        )
    )
    return fig


def draw_scatter_mapbox(lat, lon, lat_ind, lon_ind):
    fig = go.Figure(
        go.Scattermapbox(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(
                size=12,
                color=[
                    "yellow" if lat_i == lat_ind and lon_i == lon_ind else "darkred"
                    for lat_i, lon_i in zip(lat, lon)
                ],
            ),
        )
    )
    fig.update_layout(
        mapbox=dict(
            style="carto-positron", center=dict(lat=lat_ind, lon=lon_ind), zoom=8
        )
    )
    return fig


def draw_plotly_graph_spectra1d(freq, spec, dirm, spr):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=freq, y=spec, mode="lines", name="Spec (m<sup>2</sup>s)"),
        secondary_y=False,
    )
    if dirm is not None:
        fig.add_trace(
            go.Scatter(
                x=freq,
                y=dirm,
                name="dirm (deg)",
                mode="lines",
                line=dict(color="green"),
            ),
            secondary_y=True,
        )
        if spr is not None:
            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=dirm - spr,
                    name="spr- (deg)",
                    line=dict(color="red", dash="dash"),
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=dirm + spr,
                    name="spr+ (deg)",
                    line=dict(color="red", dash="dash"),
                ),
                secondary_y=True,
            )
    fig.update_yaxes(secondary_y=True, showgrid=False)
    return fig


def draw_plotly_graph_spectra(freq, spec, dirs, cmax, cmin):

    fig = go.Figure(
        go.Barpolar(
            r=freq.repeat(len(dirs)),
            theta=np.tile(dirs, len(freq)),
            width=[14.7] * len(np.tile(dirs, len(freq))),
            marker=dict(
                color=spec.flatten(),
                colorscale="Blues",
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title="m<sup>2</sup>s",
                    ticks="outside",
                    len=0.75,
                ),
            ),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            angularaxis=dict(visible=True, rotation=90, direction="clockwise"),
        ),
    )
    return fig


def open_browser(port):
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new(f"http://127.0.0.1:{port}/")


def waveseries_plotter_basic(model):
    ts = model["waveseries"]
    fig = go.Figure()

    variables = [col for col in ts.core.data_vars()]

    for variable in variables:
        trace = go.Scatter(
            x=ts.get("time"),
            y=ts.get(variable),
            mode="lines",
            name=variable,
            visible="legendonly",
        )
        fig.add_trace(trace)

    fig.update_layout(title=f"{ts.name}", xaxis_title="UTC", yaxis_title="Values")
    fig.show()


def waveseries_plotter_dash(model):
    ts = model["waveseries"]
    var = xarray_to_dataframe(ts)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1(id="title", style={"textAlign": "center"}),
            html.P("Select variable:"),
            dcc.Dropdown(
                id="waveseries-1",
                options=[{"label": val, "value": val} for val in ts.core.data_vars()],
                value="hs",
                clearable=False,
                style={"width": "30%"},
            ),
            dcc.Dropdown(
                id="waveseries-2",
                options=[{"label": "None", "value": "None"}]
                + [{"label": val, "value": val} for val in ts.core.data_vars()],
                value="None",
                clearable=False,
                style={"width": "30%"},
            ),
            html.Div(
                [
                    dcc.Graph(id="waveseries_chart"),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "width": "75",
                    "float": "left",
                },
            ),
            html.Div(
                [dcc.Graph(id="map")],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "width": "25%",
                    "float": "right",
                },
            ),
        ]
    )

    @app.callback(
        Output("waveseries_chart", "figure"),
        Output("title", "children"),
        Output("map", "figure"),
        Input("waveseries-1", "value"),
        Input("waveseries-2", "value"),
    )
    def display_time_series(var1, var2):
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        fig = px.line(var, x="time", y=var1)
        subfig.add_trace(fig.data[0], secondary_y=False)
        if var2 != "None":
            fig2 = px.line(var, x="time", y=var2)
            subfig.add_trace(fig2.data[0], secondary_y=True)
            subfig.update_traces(line_color="blue", secondary_y=False)
            subfig.update_traces(line_color="red", secondary_y=True)
            subfig.update_xaxes(minor=dict(ticks="inside", showgrid=True))
            subfig.update_yaxes(secondary_y=True, showgrid=False)
            subfig.update_layout(xaxis_title="UTC", yaxis_title=var1)
            subfig.update_yaxes(title_text=var2, secondary_y=True)
        else:
            subfig.update_layout(xaxis_title="UTC", yaxis_title=var1)
        subfig.update_layout(
            width=1300,
            height=900,
            margin=dict(l=0, r=0, t=50, b=50),
        )
        fig = go.Figure(
            go.Scattermapbox(
                lat=ts.lat(),
                lon=ts.lon(),
                mode="markers",
                marker=dict(size=12),
            )
        )
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=int(ts.lat()), lon=int(ts.lon())),
                zoom=6,
            ),
            width=450,
            height=850,
            margin=dict(l=0, r=0, t=50, b=50),
        )
        title = f"{ts.name} Waveseries"
        return subfig, title, fig

    port = random.randint(1000, 9999)
    Timer(1, open_browser, args=[port]).start()
    app.run(debug=False, port=port)


def waveseries_plotter(model, use_dash: bool):
    if use_dash:
        waveseries_plotter_dash(model)
    else:
        waveseries_plotter_basic(model)


def create_spectra_app_layout(
    len_of_inds: int, len_of_times: int, number_of_plots: int
):

    if number_of_plots == 1:
        spectral_graphs = [dcc.Graph(id="primary_graph")]
    elif number_of_plots == 2:
        spectral_graphs = [
            dcc.Graph(id="primary_graph"),
            dcc.Graph(id="secondary_graph"),
        ]
    else:
        raise ValueError("'number_of_plots' must be 1 or 2, not {number_of_plots}!")
    return html.Div(
        [
            html.H1(id="title", style={"textAlign": "center"}),
            html.H2(id="smaller_title", style={"textAlign": "center"}),
            html.Label("time_index"),
            dcc.Slider(
                min=0,
                max=len_of_times - 1,
                step=1,
                value=0,
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="drag",
                persistence=True,
                persistence_type="session",
                id="time_slider",
            ),
            html.Label("inds_index"),
            dcc.Slider(
                min=0,
                max=len_of_inds - 1,
                step=1,
                value=0,
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="drag",
                persistence=True,
                persistence_type="session",
                id="inds_slider",
            ),
            html.Div(
                spectral_graphs,
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "width": "50%",
                    "float": "left",
                },
            ),
            html.Div(
                [dcc.Graph(id="spectra_map")], style={"width": "50%", "float": "right"}
            ),
        ]
    )


def spectra_plotter(model):
    spectra = model.spectra()
    spectra1d = model.spectra1d()

    number_of_plots = 0
    if spectra1d is not None:
        number_of_plots += 1
        primary_object = "spectra1d"

    if spectra is not None:
        number_of_plots += 1
        primary_object = "spectra"

    lons, lats = model[primary_object].lonlat()
    times = model[primary_object].time(datetime=False)
    name = model[primary_object].name

    app = Dash(__name__)

    app.layout = create_spectra_app_layout(
        len_of_inds=len(lons),
        len_of_times=len(times),
        number_of_plots=number_of_plots,
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
        [Input("time_slider", "value"), Input("inds_slider", "value")],
    )
    def display_spectra(time_r, inds_r):
        spectra_map = draw_scatter_mapbox(
            lat=lats,
            lon=lons,
            lat_ind=lats[inds_r],
            lon_ind=lons[inds_r],
        )
        spectra_map.update_layout(
            width=800, height=800, margin=dict(l=50, r=0, t=10, b=50)
        )

        graphs = {}
        if spectra is not None:
            spec1 = spectra.spec(squeeze=False)[:, inds_r, :, :].flatten()
            graphs["spectra"] = draw_plotly_graph_spectra(
                freq=spectra.freq(),
                spec=spectra.spec(squeeze=False)[time_r, inds_r, :, :].flatten(),
                dirs=spectra.dirs(),
                cmin=np.min(spec1),
                cmax=np.max(spec1),
            )
            graphs["spectra"].update_layout(
                width=800,
                height=800,
                margin=dict(l=200, r=0, t=100, b=50),
            )

        if spectra1d is not None:
            spec1d = spectra1d.spec(squeeze=False)[:, inds_r, :].flatten()
            graphs["spectra1d"] = draw_plotly_graph_spectra1d(
                freq=spectra1d.freq(),
                spec=spectra1d.spec(squeeze=False)[time_r, inds_r, :],
                dirm=(
                    spectra1d.dirm(squeeze=False)[time_r, inds_r, :]
                    if spectra1d.dirm() is not None
                    else None
                ),
                spr=(
                    spectra1d.spr(squeeze=False)[time_r, inds_r, :]
                    if spectra1d.spr() is not None
                    else None
                ),
            )
            graphs["spectra1d"].update_layout(
                xaxis_title=f"{spectra1d.meta.get('freq').get('long_name')}",
                yaxis=dict(
                    title=f"{spectra1d.meta.get('spec').get('long_name')}\n {'E(f)'}",
                    range=[0, np.max(spec1d) * 1.1],
                ),
                yaxis2=dict(
                    title=f"{spectra1d.meta.get('dirm').get('long_name')}\n ({spectra1d.meta.get('dirm').get('unit')})",
                    overlaying="y",
                    side="right",
                    range=[0, np.max(spectra1d.dirm()) * 1.1],
                ),
                width=800,
                height=500,
                margin=dict(l=100, r=0, t=100, b=50),
            )

        title = f"{times[time_r]} {name}"
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


def scatter_plotter(model, model1):
    xmodel = sanitation.force_to_ds(model)
    ymodel = sanitation.force_to_ds(model1)
    xdf = sanitation.xarray_to_dataframe(xmodel)
    ydf = sanitation.xarray_to_dataframe(ymodel)

    df = pd.merge(
        xdf.set_index("time").add_suffix(f" {xmodel.name}").reset_index(),
        ydf.set_index("time").add_suffix(f" {ymodel.name}").reset_index(),
        on="time",
    )
    first_column = df.pop("time")
    df.insert(0, "time", first_column)
    df_column = [col for col in df.columns if col.endswith(f" {xmodel.name}")]
    df1_column = [col for col in df.columns if col.endswith(f" {ymodel.name}")]
    df_noNa = df.dropna().reset_index(drop=True)
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1(xmodel.name, style={"textAlign": "center"}),
            html.P("Select variable:"),
            dcc.Dropdown(
                id="x-axis-dropdown",
                options=[{"label": col, "value": col} for col in df_column],
                value=f"hs {xmodel.name}",
                clearable=False,
                style={"width": "30%"},
            ),
            dcc.Dropdown(
                id="y-axis-dropdown",
                options=[{"label": col, "value": col} for col in df1_column],
                value=f"hs {ymodel.name}",
                clearable=False,
                style={"width": "30%"},
            ),
            dcc.Graph(id="scatter_graph"),
        ]
    )

    @app.callback(
        Output("scatter_graph", "figure"),
        Input("x-axis-dropdown", "value"),
        Input("y-axis-dropdown", "value"),
    )
    def update_graph(xvar, yvar):
        xdata, ydata = df_noNa[xvar].values, df_noNa[yvar].values
        RMSE = np.sqrt(np.mean((xdata-ydata)**2))
        R = np.corrcoef(xdata, ydata)[0,1]
        SI = RMSE / np.mean(xdata)*100
        xy = np.vstack([xdata, ydata])
        z = gaussian_kde(xy)(xy)

        if xvar not in df.columns or yvar not in df.columns:
            return go.Figure()
        
        # Add scatter
        xunit = sanitation.get_units(xmodel,xvar.split(' ')[0])
        yunit = sanitation.get_units(ymodel,yvar.split(' ')[0])
        fig = px.scatter(
           df_noNa, x=xvar, y=yvar, color=z, color_continuous_scale='blues', labels={
                xvar: f"{xvar} ({xunit})",
                yvar: f"{yvar} ({yunit})"}
        )

        linear_regression_line(xdata, ydata, fig)

        # Lines
        x_values = np.linspace(0, np.ceil(np.max(xdata)), 100)
       
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

        

        xvarname = sanitation.get_varname(xmodel,xvar.split(' ')[0])
        yvarname = sanitation.get_varname(ymodel,yvar.split(' ')[0])
        text = [f"N={len(xdf)}"]
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
        fig.update_layout(width=1800, height=900, margin=dict(l=0, r=0, t=40, b=0))
        return fig

    port = random.randint(1000, 9999)
    Timer(1, open_browser, args=[port]).start()
    app.run(debug=False, port=port)
