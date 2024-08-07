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
    var = ts.ds().to_dataframe()
    var=var.reset_index()
    drop=['lon','lat','inds']
    var= var.drop(drop,axis='columns')

    fig = go.Figure()

    variables = [col for col in var.columns if col != 'time']

    for variable in variables:
        trace = go.Scatter(x=var['time'], y=var[variable], mode='lines', name=variable)
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

#install numpy scikit-learn statsmodels
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
def scatter_plotter(model: ModelRun, model1:ModelRun):
    e39 = model.waveseries()
    nora3=model1.waveseries()
    e39_ds={
        'time':e39.time(),
        'hs': e39.hs(),
        'hs_max': e39.hs_max(),
        'tm01': e39.tm01(),
        'tm02': e39.tm02(),
        'tp': e39.tp(),
        'tz': e39.tz(),
        'dirm_sea': e39.dirm_sea(),
        'dirs': e39.dirs(),
    }
    nora3_ds={
        'time':nora3.time(),
        'hs':nora3.hs(),
        'tm01':nora3.tm01(),
        'tm02':nora3.tm02(),
        'tm_10':nora3.tm_10(),
        'dirm':nora3.dirm(),
    }

    e39_df=pd.DataFrame(e39_ds)
    nora3_df=pd.DataFrame(nora3_ds)
    e39_df.set_index('time', inplace=True)
    e39_df=e39_df.resample('h').asfreq()
    e39_df=e39_df.reset_index()
    common_columns = list(set(e39_df.columns).intersection(set(nora3_df.columns)))
    df = pd.merge(e39_df[common_columns], nora3_df[common_columns], on='time', suffixes=('_e39','_nora3'))
    first_column=df.pop('time')
    df.insert(0,'time',first_column)
    app = Dash(__name__)
    app.layout = html.Div([
        html.H1(nora3.name, style={'textAlign': 'center'}),
        dcc.Dropdown(
            id="x-axis-dropdown",
            options=[
                {"label": "hs", "value": "hs"},
                {"label": "tm01", "value": "tm01"},
                {"label": "tm02", "value": "tm02"},
            ],
            value="hs",
            clearable=False,
            style={'width': '30%'}, 
        ),
        dcc.Dropdown(
            id="y-axis-dropdown",
            options=[
                {"label": "hs", "value": "hs"},
                {"label": "tm01", "value": "tm01"},
                {"label": "tm02", "value": "tm02"},
            ],
            value="hs",
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
        x_col=f'{x_var}_e39'
        y_col=f'{y_var}_nora3'
        df_noNa=df.dropna().reset_index(drop=True)
        """
        Calculates the correlation
        """
        x_mean=df_noNa[x_col].mean()
        y_mean=df_noNa[y_col].mean()
        cov_sum=0
        n=df_noNa[x_col]
        for i in range(len(n)):
            x_dev=df_noNa[x_col].iloc[i]-x_mean
            y_dev=df_noNa[y_col].iloc[i]-y_mean
            cov_sum=cov_sum + (x_dev*y_dev)
        covariance = cov_sum / len(n)
        x_varsum=0
        y_varsum=0
        for i in range(len(n)):
            x=(df_noNa[x_col].iloc[i]-x_mean)**2
            y=(df_noNa[y_col].iloc[i]-y_mean)**2
            x_varsum=x_varsum + x
            y_varsum=y_varsum + y
        x_var = x_varsum/len(n)
        y_var = y_varsum/len(n)
        x_std = (x_var)**0.5
        y_std = (y_var)**0.5
        correlation = covariance/(x_std*y_std)
        """
        Calculates RMSE
        Calculates SI
        Makes a linear regression line
        """
        if x_col not in df.columns or y_col not in df.columns:
            return go.Figure()
        X = df_noNa[[x_col]].values.reshape(-1,1)
        model=LinearRegression()
        model.fit(X,df_noNa[[y_col]])
        a=model.coef_[0][0]
        b=model.intercept_[0]
        y_estimated=(a*df_noNa[x_col]+b)
        y_rmse=0
        for i in range(len(n)):
            y=(df_noNa[y_col].iloc[i]-y_estimated.iloc[i])**2
            y_rmse=y_rmse + y
        RMSE=(y_rmse/len(n))**0.5

        SI=RMSE/x_mean

        xy = np.vstack([df_noNa[x_col].values, df_noNa[y_col].values])
        z = gaussian_kde(xy)(xy)
        x_range=np.linspace(X.min(),X.max(),100)
        y_range=model.predict(x_range.reshape(-1,1))
        fig = px.scatter(df_noNa, x=x_col, y=y_col,color=z, color_continuous_scale='jet')
        fig.add_traces(go.Scatter(x=x_range.flatten(), y=y_range.flatten(), name='Linear regression'))
        fig.update_layout(
            coloraxis_colorbar=dict(
                title='Density',
                y=0.45,
                x=1.015,
            ),
            annotations=[
                dict(
                    x=0.001,
                    y=0.995,
                    xref='paper',
                    yref='paper',
                    text=(
                        f'N = {len(df[x_col])}<br>'
                        f'Bias = {x_mean - y_mean:.4f}<br>'
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
        return fig
    app.run_server("0.0.0.0", 8000, debug=True)

