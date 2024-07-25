from __future__ import annotations
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dnora.modelrun import ModelRun

def waveseries_plotter(model: ModelRun):
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
        [Input("waveseries-1", "value"), 
         Input("waveseries-2", "value")]
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

    app.run_server("0.0.0.0", 8000, debug=True)

def spectra1d_plotter(model: ModelRun):
    spectra1d = model.spectra1d()
    
    time = {
        'time': spectra1d.time(),
    }
    inds = {
        'inds': spectra1d.inds(),
    }
    
    time_df = pd.DataFrame(time)
    time_df['time'] = pd.to_datetime(time_df['time'])
    time_df['hour'] = time_df['time'].dt.hour
    
    inds_df = pd.DataFrame(inds)
    
    app = Dash(__name__)
    
    app.layout = html.Div([
        html.H1(spectra1d.name, style={'textAlign': 'center'}),
        
        html.Label("time_index"),
        dcc.Slider(
            min=time_df['hour'].min(),
            max=time_df['hour'].max(),
            step=1,
            value=time_df['hour'].min(),
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag',
            persistence=True,
            persistence_type='session',
            id='time_slider',
        ),
        
        html.Label("inds_index"),
        dcc.Slider(
            min=inds_df['inds'].min(),
            max=inds_df['inds'].max(),
            step=1,
            value=inds_df['inds'].min(),
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag',
            persistence=True,
            persistence_type='session',
            id='inds_slider',
        ),
        
        dcc.Graph(id="spectra1d_graph"),
    ])
    
    @app.callback(
        Output("spectra1d_graph", "figure"), 
        Input("time_slider", "value"),
        Input("inds_slider", "value"),
    )
    def display_spectra1d(time_r, inds_r):
        selected_time_df = time_df[time_df["hour"] == time_r]
        spectrum = spectra1d.spec()[selected_time_df.index[0], inds_r, :]
        freqs = spectra1d.freq()
        
        fig = px.line(x=freqs, y=spectrum, labels={'x': 'Frequency', 'y': 'Wavespectrum'})
        fig.update_layout(
            yaxis_range=[0, 75],
        )
        return fig
    
    app.run_server("0.0.0.0", 8000, debug=True)
