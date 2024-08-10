from __future__ import annotations
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import os
from threading import Timer
import webbrowser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dnora.modelrun import ModelRun

def waveseries_plotter(model: ModelRun):
    ts = model.waveseries()
    var = ts.ds().to_dataframe()
    var=var.reset_index()
    drop=['lon','lat','inds']
    var= var.drop(drop,axis='columns')

    fig = go.Figure()

    variables = [col for col in var.columns if col != 'time']

    for variable in variables:
        trace = go.Scatter(x=var['time'], y=var[variable], mode='lines', name=variable, visible='legendonly')
        fig.add_trace(trace)

    fig.update_layout(
        title='Waveseries',
        xaxis_title='UTC',
        yaxis_title='Values'
    )

    fig.show()

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

def xarray_to_dataframe(model) -> pd.DataFrame:
    df=model.ds().to_dataframe()
    df=df.reset_index()
    col_drop=['lon','lat','inds']
    df=df.drop(col_drop,axis='columns')
    df.set_index('time', inplace=True)
    df=df.resample('h').asfreq()
    df=df.reset_index()
    return df
def calculate_correlation(x,y):
    x_mean = x.mean()
    y_mean = y.mean()
    covariance = ((x - x_mean) * (y - y_mean)).mean()
    x_var = ((x - x_mean) ** 2).mean()
    y_var = ((y - y_mean) ** 2).mean()
    x_std = x_var ** 0.5
    y_std = y_var ** 0.5
    correlation = covariance / (x_std * y_std)
    
    return correlation

def calculate_RMSE(x,y):
    X = x.values.reshape(-1,1)
    linear=LinearRegression()
    linear.fit(X,y)
    a=linear.coef_[0]
    b=linear.intercept_
    y_estimated=(a*x+b)
    y_rmse=(y-y_estimated)**2
    RMSE=(y_rmse.mean())**0.5
    return RMSE

def linear_regression_line(x,y,fig):
    X = x.values.reshape(-1,1)
    linear=LinearRegression()
    linear.fit(X,y)
    x_range=np.linspace(0,np.ceil(X.max()),100)
    y_range=linear.predict(x_range.reshape(-1,1))
    fig.add_traces(go.Scatter(x=x_range.flatten(), y=y_range.flatten(), mode='lines', name='Linear regression',visible=True))
    return fig

def scatter_plotter(model: ModelRun, model1:ModelRun):
    ds_model=model.waveseries()
    ds1_model1=model1.waveseries()
    df_model=xarray_to_dataframe(model.waveseries())
    df1_model1=xarray_to_dataframe(model1.waveseries())

    common_columns = list(set(df_model.columns).intersection(set(df1_model1.columns)))
    df = pd.merge(df_model[common_columns], df1_model1[common_columns], on='time', suffixes=(f' {ds_model.name}',  f' {ds1_model1.name}'))
    first_column=df.pop('time')
    df.insert(0,'time',first_column)
    df_column=[col for col in df.columns if col.endswith(f' {ds_model.name}')]
    df1_column=[col for col in df.columns if col.endswith(f' {ds1_model1.name}')]
    df_noNa=df.dropna().reset_index(drop=True)
    app = Dash(__name__)
    app.layout = html.Div([
        html.H1(ds_model.name, style={'textAlign': 'center'}),
        html.P("Select variable:"),
        dcc.Dropdown(
            id="x-axis-dropdown",
            options = [{'label': col, 'value': col} for col in df_column],
            value=f'hs {ds_model.name}',
            clearable=False,
            style={'width': '30%'}, 
        ),
        dcc.Dropdown(
            id="y-axis-dropdown",
            options=[{'label': col, 'value': col} for col in df1_column],
            value=f'hs {ds1_model1.name}',
            clearable=False,
            style={'width': '30%'},
        ),
        dcc.Graph(id="scatter_graph"),
    ])
    
    @app.callback(
        Output("scatter_graph", "figure"),
        Input("x-axis-dropdown", "value"),
        Input("y-axis-dropdown", "value"), 
    )
    def update_graph(x_var,y_var):
        x_col=f'{x_var}'
        y_col=f'{y_var}'
        """
        Calculates the correlation
        """
        correlation=calculate_correlation(df_noNa[x_col],df_noNa[y_col])
        """
        Calculates RMSE
        Calculates SI
        """
        RMSE=calculate_RMSE(df_noNa[x_col],df_noNa[y_col])
        SI=RMSE/df_noNa[x_col].mean()
        """
        Stack values and
        Calculates density.
        """
        xy = np.vstack([df_noNa[x_col].values, df_noNa[y_col].values])
        z = gaussian_kde(xy)(xy)

        if x_col not in df.columns or y_col not in df.columns:
            return go.Figure()
        fig = px.scatter(df_noNa, x=x_col, y=y_col,color=z, color_continuous_scale='jet')

        linear_regression_line(df_noNa[x_col],df_noNa[y_col],fig)

        x_max=np.ceil(df_noNa[x_col].max())
        y_max=np.ceil(df_noNa[y_col].max())

        x_values = np.linspace(0, np.ceil(x_max), 100)
        y_values = x_values
        fig.add_traces(go.Scatter(x=x_values, y=y_values, mode='lines', name='x=y',visible='legendonly'))

        x_line=np.linspace(0,np.ceil(x_max), 100)
        a=np.sum(df_noNa[x_col]*df_noNa[y_col])/np.sum(df_noNa[x_col]**2)
        y=a*x_line
        fig.add_traces(go.Scatter(x=x_line, y=y, mode='lines', name='one-parameter-linear regression',visible='legendonly'))

        if x_max > y_max:
            fig.update_layout(
                yaxis=dict(range=[0, x_max]),
                xaxis=dict(range=[0, x_max])
            )
        else: 
            fig.update_layout(
                xaxis=dict(range=[0, y_max]),
                yaxis=dict(range=[0, y_max])
            )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title='Density',
                y=0.45, x=1.015,
                len=0.9
            ),
            annotations=[
                dict(
                    x=0.001,y=0.995,
                    xref='paper',
                    yref='paper',
                    text=(
                        f'N = {len(df[x_col])}<br>'
                        f'Bias = {df_noNa[x_col].mean() - df_noNa[y_col].mean():.4f}<br>'
                        f'R\u00b2= {correlation:.4f}<br>'
                        f'RMSE= {RMSE:.4F}<br>'
                        f'SI= {SI:.4F}'
                    ),
                    showarrow=False,
                    font=dict(size=16, color='black'),
                    align='left',
                    bgcolor='white',
                    borderpad=4,
                    bordercolor='black',
                    opacity=0.55
                )
            ]
        )
        fig.update_layout(
            width=1800,
            height=900,
            margin=dict(
                l=0,r=0,t=40,b=0
            )
        )
        
        return fig
    
    def open_browser():
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new('http://127.0.0.1:1222/')

    Timer(1, open_browser).start()
    app.run_server(debug=True, port=1222)