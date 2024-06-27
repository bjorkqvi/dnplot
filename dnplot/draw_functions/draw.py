import numpy as np
from ..defaults import default_markers
import matplotlib.tri as mtri
from cartopy import feature as cfeature


def draw_gridded_magnitude(
    fig_dict, x, y, data, cmap, vmax=None, vmin=None, label: str = "__nolegend__"
):
    """
    Takes lon and lat positions, and data points. Plots countourplot on given ax.
    """
    fig = fig_dict.get("fig")
    ax = fig_dict.get("ax")
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    # if vmax-vmin<20:
    levels = np.linspace(
        np.floor(vmin), np.ceil(vmax), np.floor(vmax - vmin + 2).astype(int)
    )
    # else:
    #     levels = np.linspace(np.floor(vmin), np.ceil(vmax), 11)
    xx, yy = np.meshgrid(x, y)
    tri = mtri.Triangulation(xx.ravel(), yy.ravel())
    if len(levels) > 1:
        cont = ax.tricontourf(tri, data.ravel(), cmap=cmap, levels=levels)
    else:
        cont = ax.pcolor(x, y, data, cmap=cmap, label=label)

    cbar = fig_dict.get("cbar") or fig.colorbar(cont)

    fig_dict["cbar"] = cbar
    fig_dict["cont"] = cont

    return fig_dict


def draw_masked_points(
    fig_dict,
    grid,
    masks_to_plot: list[str],
    default_dict: dict = default_markers.get("generic_masks"),
):
    """Plots the masked points to the plot"""
    for mask in masks_to_plot:
        markers = dict(default_dict)
        markers.update(default_markers.get(mask, default_dict))

        x, y = grid.xy(native=True, mask=grid.get(mask))
        fig_dict.get("ax").plot(
            x,
            y,
            markers.get("marker") + markers.get("color"),
            markersize=markers.get("size"),
            label="Set " + mask.split("_")[0] + " points",
        )

    return fig_dict


def draw_object_points(
    fig_dict,
    model,
    objects_to_plot: list[str],
    default_dict: dict = default_markers.get("generic_objects"),
):
    """Plots points og objects to the plot"""
    for obj_type in objects_to_plot:
        markers = dict(default_dict)
        markers.update(default_markers.get(obj_type.name.lower(), default_dict))

        if model[obj_type] is not None:
            x, y = model[obj_type].xy(native=True)
            fig_dict.get("ax").plot(
                x,
                y,
                markers.get("marker") + markers.get("color"),
                markersize=markers.get("size"),
                label="Imported " + obj_type.name.lower(),
            )

    return fig_dict


def draw_mask(fig_dict, x_vec, y_vec, mask) -> dict:
    xx, yy = np.meshgrid(x_vec, y_vec)
    mask = mask.astype(float)

    mask[mask < 1] = np.nan

    fig_dict.get("ax").contourf(xx, yy, mask, cmap="gray")

    return fig_dict


def draw_arrows(fig_dict, lon, lat, xdata, ydata, scale=100, reduce_arrows: int = None):
    """

    Parameters
    ----------
    ax : ax
        DESCRIPTION.
    lon : np.array or alike. len=m
        DESCRIPTION.
    lat : np.array or alike. len=n
        DESCRIPTION.
    xdata : gridded data [mxn]
        DESCRIPTION.
    ydata : gridded data [mxn]
        DESCRIPTION.
    reduce_arrows : int, optional
        The default is 2. Higher numer=reduce number of arrows more
    scale : int, optional
        Scale length of the arrow, the width and the length of arrow head. The default is 100.

    Returns
    -------
    ax : ax, with arrows

    """
    if reduce_arrows is not None:
        step_lat = reduce_arrows
        step_lon = reduce_arrows
    else:
        step_lat = max(round(len(lat) / 10), 1)
        step_lon = max(round(len(lon) / 10), 1)

    ax = fig_dict.get("ax")
    for m in range(0, len(lon), step_lon):
        for n in range(0, len(lat), step_lat):
            ax.arrow(
                lon[m],
                lat[n],
                xdata[n][m] / scale,
                ydata[n][m] / scale,
                color="white",
                linewidth=0.15,
                head_width=2 / scale,
                head_length=2 / scale,
                overhang=1,
            )  # linewidth=.02, head_width=.01, head_length=.01
    return fig_dict


def draw_coastline(fig_dict) -> dict:
    ax = fig_dict.get("ax")
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "coastline", "10m", facecolor="none", edgecolor="black"
        )
    )
    return fig_dict


def draw_polar_spectra(fig_dict, spec, freq, dirs) -> dict:
    fig = fig_dict.get("fig")
    ax = fig_dict.get("ax")

    last_row = np.transpose([spec[:, 0]])
    spec_plot = np.hstack([spec, last_row])
    dir_plot = np.hstack([dirs, dirs[0] + 360])
    vmin = np.min(spec)
    vmax = np.max(spec)
    # if vmax-vmin<20:
    #     levels = np.linspace(vmin, vmax, np.floor(vmax-vmin+1).astype(int))
    # # else:
    levels = np.linspace(vmin, vmax, 20)

    cont = ax.contourf(
        np.deg2rad(dir_plot), freq, spec_plot, cmap="ocean_r", levels=levels
    )
    fig_dict["cont"] = cont
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # ax.set_rticks([0, 45, 90, 135, 180, 225, 270, 315])
    # ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_ylabel("")
    ax.set_xlabel("")
    # ax.set_title(title_str)

    # orientation = 'vertical'

    # if cbar:
    #     cax = fig.add_axes([ax.get_position().x1+0.04,ax.get_position().y0,0.02,ax.get_position().height])
    #     cbar = fig.colorbar(cont, orientation=orientation, cax=cax, label=f"E(f, theta) (m**2/Hz/rad)")
    #     fig_dict['cbar'] = cbar
    #     fig_dict['cax'] = cax
    return fig_dict
