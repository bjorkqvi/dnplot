from ..draw_functions import draw_matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from ..defaults import default_variable
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats import gaussian_kde
import xarray as xr


def grid_plotter(fig_dict: dict, model) -> dict:
    """Plot the depth information and set output points etc."""
    grid = model["grid"]
    fig_dict = draw_matplotlib.draw_gridded_magnitude(
        fig_dict,
        grid.x(native=True),
        grid.y(native=True),
        grid.topo(),
        cmap=default_variable["topo"]["cmap"],
    )
    if hasattr(grid, "land_mask"):
        fig_dict = draw_matplotlib.draw_mask(
            fig_dict, grid.x(native=True), grid.y(native=True), grid.land_mask()
        )

    fig_dict["ax"].set_xlabel(grid.core.x_str)
    fig_dict["ax"].set_ylabel(grid.core.y_str)
    fig_dict["cbar"].set_label("Depth [m]")
    fig_dict["ax"].set_title(f"{grid.name} {grid.ds().attrs.get('source', '')}")

    # masks_to_plot = ["spectra_mask", "output_mask"]
    # fig_dict = draw_matplotlib.draw_masked_points(fig_dict, grid, masks_to_plot=masks_to_plot)

    objects_to_plot = [
        "wind",
    ]
    fig_dict = draw_matplotlib.draw_object_points(
        fig_dict, model, objects_to_plot=objects_to_plot
    )

    fig_dict.get("ax").legend()

    return fig_dict


def wind_plotter(fig_dict: dict, model) -> dict:
    def update_plot(val):
        nonlocal fig_dict
        nonlocal figure_initialized
        fig_dict = draw_matplotlib.draw_gridded_magnitude(
            fig_dict,
            wind.x(native=True),
            wind.y(native=True),
            wind.mag()[val, :, :],
            vmax=np.max(wind.mag()),
            vmin=0,
            cmap=default_variable["ff"]["cmap"],
        )
        fig_dict = draw_matplotlib.draw_coastline(fig_dict)
        fig_dict = draw_matplotlib.draw_arrows(
            fig_dict,
            wind.x(native=True),
            wind.y(native=True),
            wind.u()[val, :, :],
            wind.v()[val, :, :],
        )
        # if not figure_initialized:
        #     masks_to_plot = ["output_mask"]
        #     fig_dict = draw_matplotlib.draw_masked_points(fig_dict, grid, masks_to_plot=masks_to_plot)
        #     fig_dict.get("ax").legend()
        fig_dict["ax"].set_title(f"{wind.time(datetime=False)[val]} {wind.name}")
        figure_initialized = True

    wind = model.get("wind")
    grid = model.get("grid")
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


def spectra_plotter(fig_dict: dict, model) -> dict:
    def update_plot(val):
        nonlocal fig_dict
        nonlocal figure_initialized
        fig_dict = draw_matplotlib.draw_polar_spectra(
            fig_dict,
            spectra.spec()[sliders["time"].val, sliders["inds"].val, :, :],
            spectra.freq(),
            spectra.dirs(),
        )

        fig_dict["ax"].set_title(
            f"{spectra.time(datetime=False)[sliders['time'].val]} {spectra.name} \n Latitude={spectra.lat()[sliders['inds'].val]:.4f} Longitude={spectra.lon()[sliders['inds'].val]:.4f}"
        )
        figure_initialized = True

    spectra = model.get("spectra")
    grid = model.get("grid")
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


def waveseries_plotter(model, var: list[str]):
    ts = model.get("waveseries")
    if len(var) < 4:
        fig, axes = plt.subplots(len(var), 1)
        fig.suptitle(ts.name, fontsize=16)
        axes = axes if len(var) > 1 else [axes]
        for i, item in enumerate(var):
            if isinstance(item, tuple):
                var1, var2 = item
                ax = axes[i]
                ax.plot(
                    ts.get("time"),
                    ts.get(var1),
                    color="b",
                    label=f"{var1} ({ts.meta.get(var1)['unit']})",
                )

                ax.set_ylabel(
                    f"{ts.meta.get(var1)['long_name']}\n ({ts.meta.get(var1)['unit']})",
                    color="b",
                )
                ax.set_xlabel("UTC", fontsize=12)
                ax2 = ax.twinx()
                ax2.plot(
                    ts.get("time"),
                    ts.get(var2),
                    color="g",
                    label=f"{var2} ({ts.meta.get(var2)['unit']})",
                )
                ax2.set_ylabel(
                    f"{ts.meta.get(var2)['long_name']}\n ({ts.meta.get(var2)['unit']})",
                    color="g",
                )
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
                ax.grid(True)

            else:
                axes[i].plot(
                    ts.get("time"),
                    ts.get(item),
                    color="b",
                    label=f"{item} ({ts.meta.get(item)['unit']})",
                )
                axes[i].set_ylabel(
                    f"{ts.meta.get(item)['long_name']} \n ({ts.meta.get(item)['unit']})"
                )
                axes[i].set_xlabel("UTC", fontsize=12)
                axes[i].legend()
                axes[i].grid(True)
        plt.tight_layout()
        plt.show()

    else:
        for item in var:
            fig, ax = plt.subplots()
            if isinstance(item, tuple):
                var1, var2 = item
                ax.plot(
                    ts.get("time"),
                    ts.get(var1),
                    color="b",
                    label=f"{var1} ({ts.meta.get(var1)['unit']})",
                )

                ax.set_ylabel(
                    f"{ts.meta.get(var1)['long_name']}\n ({ts.meta.get(var1)['unit']})",
                    color="b",
                )
                ax.set_xlabel("UTC", fontsize=12)
                ax2 = ax.twinx()
                ax2.plot(
                    ts.get("time"),
                    ts.get(var2),
                    color="g",
                    label=f"{var2} ({ts.meta.get(var2)['unit']})",
                )
                ax2.set_ylabel(
                    f"{ts.meta.get(var2)['long_name']}\n ({ts.meta.get(var2)['unit']})",
                    color="g",
                )
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2)
                ax.grid(True)
            else:
                ax.plot(
                    ts.get("time"),
                    ts.get(item),
                    color="b",
                    label=f"{item} ({ts.meta.get(item)['unit']})",
                )
                ax.set_xlabel("UTC", fontsize=12)
                ax.set_ylabel(
                    f"{ts.meta.get(item)['long_name']} \n ({ts.meta.get(item)['unit']})"
                )
                ax.legend()
                ax.grid(True)
            ax.set_title(ts.name, fontsize=16)

        plt.tight_layout()
        plt.show()


def spectra1d_plotter(fig_dict: dict, model) -> dict:
    def update_plot(val):
        nonlocal fig_dict
        nonlocal figure_initialized
        ax = fig_dict["ax"]
        ax2 = fig_dict["ax2"]
        ax.cla()
        ax2.cla()
        dirm = None
        spr = None
        if spectra1d.dirm() is not None:
            dirm = spectra1d.dirm()[sliders["time"].val, sliders["inds"].val, :]
        if spectra1d.spr() is not None:
            spr = spectra1d.spr()[sliders["time"].val, sliders["inds"].val, :]

        fig_dict = draw_matplotlib.draw_graph_spectra1d(
            fig_dict,
            spectra1d.spec()[sliders["time"].val, sliders["inds"].val, :],
            spectra1d.freq(),
            dirm,
            spr,
        )

        ax.set_ylim(0, np.max(spectra1d.spec()[:, sliders["inds"].val, :]) * 1.1)
        ax.set_title(
            f"{spectra1d.time(datetime=False)[sliders['time'].val]} {spectra1d.name} \n Latitude={spectra1d.lat()[sliders['inds'].val]:.4f} Longitude={spectra1d.lon()[sliders['inds'].val]:.4f}"
        )
        ax.set_xlabel("Frequency")
        ax.set_ylabel(
            f"{spectra1d.meta.get('spec').get('long_name')}\n {'E(f)'}", color="b"
        )
        ax2.set_ylim(0, np.max(spectra1d.dirm()) * 1.1)
        ax2.set_ylabel(
            f"{spectra1d.meta.get('dirm').get('long_name')}\n {spectra1d.meta.get('dirm').get('unit')}",
            color="g",
        )
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax.grid()
        figure_initialized = True

    spectra1d = model.get("spectra1d")
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


def scatter_plotter(fig_dict: dict, model, var):
    ts = model.get("waveseries")
    x = var[0]
    y = var[1]
    title = rf"$\bf{{{ts.name}}}$" + "\n" + rf"{x} vs {y}"
    fig_dict["ax"].set_title(title, fontsize=14)
    fig_dict["ax"].scatter(
        ts.get(x), ts.get(y), c="blue", alpha=0.6, edgecolors="w", s=100
    )
    fig_dict["ax"].set_xlabel(
        f"{ts.meta.get(x)['long_name']}\n ({ts.meta.get(x)['unit']})"
    )
    fig_dict["ax"].set_ylabel(
        f"{ts.meta.get(y)['long_name']}\n ({ts.meta.get(y)['unit']})"
    )
    fig_dict["ax"].grid(linestyle="--")
    plt.show(block=True)


def xarray_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    df = ds.to_dataframe()
    df = df.reset_index()
    col_drop = ["lon", "lat", "inds"]
    df = df.drop(col_drop, axis="columns")
    df.set_index("time", inplace=True)
    df = df.resample("h").asfreq()
    df = df.reset_index()
    return df


def calculate_correlation(x, y):
    x_mean = x.mean()
    y_mean = y.mean()
    covariance = ((x - x_mean) * (y - y_mean)).mean()
    x_var = ((x - x_mean) ** 2).mean()
    y_var = ((y - y_mean) ** 2).mean()
    x_std = x_var**0.5
    y_std = y_var**0.5
    correlation = covariance / (x_std * y_std)
    return correlation


def calculate_RMSE(x, y):
    X = x.values.reshape(-1, 1)
    linear = LinearRegression()
    linear.fit(X, y)
    a = linear.coef_[0]
    b = linear.intercept_
    y_estimated = a * x + b
    y_rmse = (y - y_estimated) ** 2
    RMSE = (y_rmse.mean()) ** 0.5
    return RMSE


def scatter1_plotter(fig_dict: dict, model, model2, var):
    ws = model.get("waveseries")
    ws2 = model2.get("waveseries")
    x = var[0]
    y = var[1]
    df = xarray_to_dataframe(ws.ds())
    df1 = xarray_to_dataframe(ws2.ds())
    combined_df = pd.concat([df, df1], axis=1)

    combined_df_cleaned = combined_df.dropna()

    df = combined_df_cleaned.iloc[:, : df.shape[1]].reset_index(drop=True)
    df1 = combined_df_cleaned.iloc[:, df.shape[1] :].reset_index(drop=True)
    correlation = calculate_correlation(df[x], df1[y])

    RMSE = calculate_RMSE(df[x], df1[y])
    SI = RMSE / df[x].mean()
    X = df[x].values.reshape(-1, 1)
    linear = LinearRegression()
    linear.fit(X, df1[y])

    x_range = np.linspace(0, np.ceil(X.max()), 100)
    y_range = linear.predict(x_range.reshape(-1, 1))
    # Text on the figure
    text = "\n".join(
        (
            f"N={len(df)}",
            f"Bias{df[x].mean() - df1[y].mean():.4f}",
            f"R\u00b2={correlation:.4f}",
            f"RMSE={RMSE:.4f}",
            f"SI={SI:.4f}",
        )
    )
    # color for scatter density
    xy = np.vstack([df[x].values, df1[y].values])
    z = gaussian_kde(xy)(xy)
    norm = Normalize(vmin=z.min(), vmax=z.max())
    cmap = cm.jet
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    title = rf"$\bf{{{ws.name}}}$" + "\n" + rf"{x} vs {y}"
    fig_dict["ax"].set_title(title, fontsize=14)
    fig_dict["ax"].scatter(df[x], df1[y], c=z, cmap=cmap, norm=norm, s=50)
    x_max = np.ceil(df[x].max())
    y_max = np.ceil(df1[y].max())

    if x_max > y_max:
        fig_dict["ax"].set_ylim([0, x_max])
        fig_dict["ax"].set_xlim([0, x_max])
    else:
        fig_dict["ax"].set_xlim([0, y_max])
        fig_dict["ax"].set_ylim([0, y_max])

    fig_dict["ax"].plot(
        x_range, y_range, color="red", linewidth=2, label="Regression line"
    )

    x_line = np.linspace(0, np.ceil(df[x].max()), 100)
    a = np.sum(df[x] * df1[y]) / np.sum(df[x] ** 2)
    y_line = a * x_line

    fig_dict["ax"].plot(x_line, y_line, linewidth=2, label="One parameter line")

    x_values = np.linspace(0, np.ceil(df[x].max()), 100)
    y_values = x_values
    fig_dict["ax"].plot(x_values, y_values, linewidth=2, label="x=y")

    fig_dict["ax"].set_xlabel(
        f"{ws.meta.get(x)['long_name']}\n ({ws.meta.get(x)['unit']})"
    )
    fig_dict["ax"].set_ylabel(
        f"{ws2.meta.get(y)['long_name']}\n ({ws2.meta.get(y)['unit']})"
    )

    # color bar
    cbar = plt.colorbar(sm, ax=fig_dict["ax"])
    cbar.set_label("Density", rotation=270, labelpad=15)

    props = dict(boxstyle="square", facecolor="white", alpha=0.6)
    ax = plt.gca()
    fig_dict["ax"].text(
        0.005,
        0.90,
        text,
        bbox=props,
        fontsize=12,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )
    fig_dict["ax"].grid(linestyle="--")
    fig_dict["ax"].legend(loc="upper left")
    plt.show(block=True)
