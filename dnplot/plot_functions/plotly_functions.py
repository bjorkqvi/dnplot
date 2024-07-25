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
