from __future__ import annotations
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dnora.modelrun import ModelRun

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

def draw_scatter_mapbox(lat,lon,lat_ind,lon_ind):
    fig=go.Figure(go.Scattermapbox(
        lat=lat,
        lon=lon,
        mode='markers',
        marker=dict(size=12, color=['yellow' if lat_i == lat_ind and lon_i == lon_ind else 'darkred' for lat_i, lon_i in zip(lat, lon)]),
    ))
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=lat_ind,lon=lon_ind),
            zoom=8
        )
    )
    return fig


def draw_plotly_graph_spectra1d(freq,spec,dirm,spr):
    fig=make_subplots(specs=[[{'secondary_y':True}]])
    fig.add_trace(
        go.Scatter(x=freq,y=spec,mode='lines',name='Spec (m<sup>2</sup>s)'),
        secondary_y=False
        )
    if dirm is not None:
        fig.add_trace(
            go.Scatter(x=freq,y=dirm,name='dirm (deg)', mode='lines',
            line = dict(color='green')),
            secondary_y=True
        )
        if spr is not None:
            fig.add_trace(
                go.Scatter(x=freq,y=dirm-spr,name='spr- (deg)',
                line = dict(color='red', dash='dash')),
                secondary_y=True
            )
            fig.add_trace(
                go.Scatter(x=freq,y=dirm+spr,name='spr+ (deg)',
                line = dict(color='red', dash='dash')),
                secondary_y=True
            )
    fig.update_yaxes(secondary_y=True, showgrid=False)
    return fig



def draw_plotly_graph_spectra(freq, spec, dirs,cmax,cmin):
    
    fig = go.Figure(go.Barpolar(
        r=freq.repeat(len(dirs)),  
        theta=np.tile(dirs, len(freq)),  
        width=[14.7] * len(np.tile(dirs, len(freq))),
        marker=dict(
            color=spec.flatten(),
            colorscale='Blues',
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title='m<sup>2</sup>s',
                ticks='outside',
                len=0.75,
            ),
        )
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                tickmode='array',
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=[0, 0.1, 0.2, 0.3, 0.4, 0.5]
            ),
            angularaxis=dict(
                visible=True,
                rotation=90,
                direction='clockwise'
            )
        ),
    )
    return fig

def open_browser(port):
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new(f'http://127.0.0.1:{port}/')


def waveseries_plotter_basic(model: ModelRun):
    ts = model.waveseries()
    var=xarray_to_dataframe(ts)
    fig = go.Figure()

    variables = [col for col in var.columns if col != 'time']

    for variable in variables:
        trace = go.Scatter(x=var['time'], y=var[variable], mode='lines', name=variable, visible='legendonly')
        fig.add_trace(trace)

    fig.update_layout(
        title=f'{ts.name}',
        xaxis_title='UTC',
        yaxis_title='Values'
    )
    fig.show()

def waveseries_plotter_dash(model: ModelRun):
    ts = model.waveseries()
    var=xarray_to_dataframe(ts)

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1(id="title", style={'textAlign': 'center'}),
        html.P("Select variable:"),
        dcc.Dropdown(
            id="waveseries-1",
            options=[
                {'label': col, 'value': col} for col in var if col!='time'
            ],
            value="hs",
            clearable=False,
            style={'width': '30%'},
        ),
        dcc.Dropdown(
            id="waveseries-2",
            options=[
                {"label": "None", "value": "None"}] +
                [{'label': col, 'value': col} for col in var if col!='time'
            ],
            value="None",
            clearable=False,
            style={'width': '30%'},
        ),
        html.Div([
            dcc.Graph(id="waveseries_chart"),
        ],style={'display': 'flex', 'flexDirection': 'column', 'width': '75', 'float': 'left'}),
        html.Div([
            dcc.Graph(id="map")
        ],style={'display': 'flex', 'flexDirection': 'column', 'width': '25%', 'float': 'right'}),

    ])

    @app.callback(
        Output("waveseries_chart", "figure"),
        Output('title','children'),
        Output("map", "figure"),
        Input("waveseries-1", "value"), 
        Input("waveseries-2", "value")
    )
    def display_time_series(var1, var2):
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        fig = px.line(var, x='time', y=var1)
        subfig.add_trace(fig.data[0], secondary_y=False)
        if var2 != "None":
            fig2 = px.line(var, x='time', y=var2)
            subfig.add_trace(fig2.data[0], secondary_y=True)
            subfig.update_traces(line_color='blue', secondary_y=False)
            subfig.update_traces(line_color='red', secondary_y=True)
            subfig.update_xaxes(minor=dict(ticks="inside", showgrid=True))
            subfig.update_yaxes(secondary_y=True, showgrid=False)
            subfig.update_layout(xaxis_title="UTC", yaxis_title=var1)
            subfig.update_yaxes(title_text=var2, secondary_y=True)
        else:
            subfig.update_layout(xaxis_title="UTC", yaxis_title=var1)
        subfig.update_layout(
            width=1300,
            height=900,
            margin=dict(
                l=0,r=0,t=50,b=50
            ),
        )
        fig=go.Figure(go.Scattermapbox(
            lat=ts.lat(),
            lon=ts.lon(),
            mode='markers',
            marker=dict(size=12),
        ))
        fig.update_layout(
            mapbox=dict(
                style='carto-positron',
                center=dict(lat=int(ts.lat()),lon=int(ts.lon())),
                zoom=6,
            ),
            width=450,
            height=850,
            margin=dict(
                l=0,r=0,t=50,b=50
            )
        )
        title = f"{ts.name} Waveseries"
        return subfig, title,fig
    port = random.randint(1000, 9999)
    Timer(1, open_browser, args=[port]).start()
    app.run_server(debug=True, port=port)

def waveseries_plotter(model: ModelRun, use_dash: bool):
        if use_dash:
            waveseries_plotter_dash(model)
        else:
            waveseries_plotter_basic(model)



def spectra_plotter(model: ModelRun):
    spectra=model.spectra()
    spectra1d = model.spectra1d()
    time = {
        'time': spectra.time(),
    }
    inds = {
        'inds': spectra.inds(),
    }
    time_df = pd.DataFrame(time)
    time_df['time'] = pd.to_datetime(time_df['time'])
    time_df['hour'] = time_df['time'].dt.hour
    inds_df = pd.DataFrame(inds)
    
    app = Dash(__name__)
    
    app.layout = html.Div([
        html.H1(id="title", style={'textAlign': 'center'}),
        html.H2(id='smaller_title', style={'textAlign': 'center'}),
        
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
        html.Div([
            dcc.Graph(id="spectra1d_graph"),
            dcc.Graph(id="spectra2d_graph")
        ],style={'display': 'flex', 'flexDirection': 'column', 'width': '50%', 'float': 'left'}),

        html.Div([
            dcc.Graph(id="spectra_map")
        ], style={'width': '50%', 'float': 'right'})
    ])
    
    @app.callback(
        [Output('title','children'),
         Output('smaller_title', 'children'),
         Output("spectra1d_graph", "figure"),
         Output('spectra_map','figure'),
         Output("spectra2d_graph", "figure")],
        [Input("time_slider", "value"),
         Input("inds_slider", "value")],
    )
    def display_spectra(time_r, inds_r):
        selected_time_df = time_df[time_df["hour"] == time_r]
        spec1=spectra.spec()[:, inds_r, :, :].flatten()
        spec1d=spectra1d.spec()[:, inds_r, :].flatten()

        fig_right=draw_plotly_graph_spectra(
            freq=spectra.freq(),
            spec=spectra.spec()[selected_time_df.index[0], inds_r, :, :].flatten(),
            dirs=spectra.dirs(),
            cmin=np.min(spec1),
            cmax=np.max(spec1),
        )
        fig_right.update_layout(
            width=900,
            height=900,
            margin=dict(
                l=200,r=0,t=100,b=50
            ),
        )

        fig_right2=draw_scatter_mapbox(
            lat=spectra.lat(),
            lon=spectra.lon(),
            lat_ind=spectra.lat()[inds_r],
            lon_ind=spectra.lon()[inds_r]
            )
        fig_right2.update_layout(
            width=1000,
            height=500,
            margin=dict(
                l=50,r=0,t=10,b=50
            )
        )

        fig_left=draw_plotly_graph_spectra1d(
            freq=spectra1d.freq(),
            spec=spectra1d.spec()[selected_time_df.index[0], inds_r, :],
            dirm=spectra1d.dirm()[selected_time_df.index[0], inds_r, :] if spectra1d.dirm() is not None else None,
            spr=spectra1d.spr()[selected_time_df.index[0], inds_r, :] if spectra1d.spr() is not None else None)
        fig_left.update_layout(
            xaxis_title=f"{spectra1d.meta.get('freq').get('long_name')}",
            yaxis=dict(
                title=f"{spectra1d.meta.get('spec').get('long_name')}\n {'E(f)'}",
                range=[0,np.max(spec1d)*1.1]
            ),
            yaxis2=dict(
                title=f"{spectra1d.meta.get('dirm').get('long_name')}\n ({spectra1d.meta.get('dirm').get('unit')})",
                overlaying='y',
                side='right',
                range=[0,np.max(spectra1d.dirm())*1.1],
            ),
            width=1000,
            height=500,
            margin=dict(
                l=100,r=0,t=100,b=50
            )
        )
        title = f"{spectra.time(datetime=False)[selected_time_df.index[0]]} {spectra.name}"
        smaller_title = f"Latitude={spectra.lat()[inds_r]:.4f} Longitude={spectra.lon()[inds_r]:.4f}"
        
        return title, smaller_title, fig_left, fig_right, fig_right2
    port = random.randint(1000, 9999)
    Timer(1, open_browser, args=[port]).start()
    app.run_server(debug=True, port=port)


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
        html.H1(id="title", style={'textAlign': 'center'}),
        html.H2(id='smaller_title', style={'textAlign': 'center'}),
        
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
        html.Div([
            dcc.Graph(id="spectra1d_graph")
            ], style={'width': '50%', 'float': 'left'}),
        html.Div([
            dcc.Graph(id="spectra_map")
        ], style={'width': '45%', 'float': 'right'})
    ])
    
    @app.callback(
        [Output('title','children'),
         Output('smaller_title', 'children'),
         Output("spectra1d_graph", "figure"),
         Output('spectra_map','figure')],
        [Input("time_slider", "value"),
         Input("inds_slider", "value")],
    )
    def display_spectra1d(time_r, inds_r):
        selected_time_df = time_df[time_df["hour"] == time_r]
        spec1d=spectra1d.spec()[:, inds_r, :].flatten()

        fig=draw_plotly_graph_spectra1d(
            freq=spectra1d.freq(),
            spec=spectra1d.spec()[selected_time_df.index[0], inds_r, :],
            dirm=spectra1d.dirm()[selected_time_df.index[0], inds_r, :] if spectra1d.dirm() is not None else None,
            spr=spectra1d.spr()[selected_time_df.index[0], inds_r, :] if spectra1d.spr() is not None else None)
        fig.update_layout(
            xaxis_title=f"{spectra1d.meta.get('freq').get('long_name')}",
            yaxis=dict(
                title=f"{spectra1d.meta.get('spec').get('long_name')}\n {'E(f)'}",
                range=[0, (np.max(spec1d)*1.1)],
            ),
            yaxis2=dict(
                title=f"{spectra1d.meta.get('dirm').get('long_name')}\n ({spectra1d.meta.get('dirm').get('unit')})",
                overlaying='y',
                side='right',
                range=[0,(np.max(spectra1d.dirm())*1.1)],
            ),
            width=1000,
            height=800,
            margin=dict(
                l=0,r=0,t=20,b=0
            )
        )
        fig1=draw_scatter_mapbox(
            lat=spectra1d.lat(),
            lon=spectra1d.lon(),
            lat_ind=spectra1d.lat()[inds_r],
            lon_ind=spectra1d.lon()[inds_r]
            )
        fig1.update_layout(
            width=700,
            height=800,
            margin=dict(
                l=50,r=0,t=10,b=50
            )
        )

        title = f"{spectra1d.time(datetime=False)[selected_time_df.index[0]]} {spectra1d.name}"
        smaller_title = f"Latitude={spectra1d.lat()[inds_r]:.4f} Longitude={spectra1d.lon()[inds_r]:.4f}"
        return title, smaller_title, fig, fig1
    
    port = random.randint(1000, 9999)
    Timer(1, open_browser, args=[port]).start()
    app.run_server(debug=True, port=port)



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
                        f'N = {len(df_noNa[x_col])}<br>'
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
    
    port = random.randint(1000, 9999)
    Timer(1, open_browser, args=[port]).start()
    app.run_server(debug=True, port=port)