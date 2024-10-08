import numpy as np
from ..defaults import default_markers
import matplotlib.tri as mtri
from cartopy import feature as cfeature
from typing import Optional


def draw_gridded_magnitude(
    fig_dict: dict,
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    cmap,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    label: str = "__nolegend__",
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

    levels = np.linspace(
        np.min(np.floor(vmin) - 1, 0),
        np.ceil(vmax) + 1,
        np.floor(vmax - vmin + 3).astype(int),
    )

    if len(levels) > 1 and not np.isclose(np.diff(levels)[0], 0):
        xx, yy = np.meshgrid(x, y)
        tri = mtri.Triangulation(xx.ravel(), yy.ravel())
        cont = ax.tricontourf(tri, data.ravel(), cmap=cmap, levels=levels)
    else:
        cont = ax.pcolor(x, y, data, cmap=cmap, label=label)

    cbar = fig_dict.get("cbar") or fig.colorbar(cont)

    fig_dict["cbar"] = cbar
    fig_dict["cont"] = cont

    return fig_dict


def draw_masked_points(
    fig_dict: dict,
    grid,
    masks_to_plot: list[str],
    default_dict: dict = default_markers.get("generic_points"),
):
    """Plots the masked points to the plot"""
    if grid is None:
        return fig_dict
    for mask in masks_to_plot:
        markers = dict(default_dict)
        markers.update(default_markers.get(f"{mask}_points", default_dict))

        if grid.get(f"{mask}_points") is not None:
            x, y = grid.get(f"{mask}_points")

            fig_dict.get("ax").plot(
                x,
                y,
                markers.get("marker") + markers.get("color"),
                markersize=markers.get("size"),
                label="Set " + mask.split("_")[0] + " points",
            )

    return fig_dict


def draw_object_points(
    fig_dict: dict,
    model: dict,
    objects_to_plot: list[str],
    default_dict: dict = default_markers.get("generic_objects"),
):
    """Plots points og objects to the plot"""
    for obj_type in objects_to_plot:
        markers = dict(default_dict)
        markers.update(default_markers.get(obj_type, default_dict))
        if model.get(obj_type) is not None:
            x, y = model[obj_type].xy(native=True)
            fig_dict.get("ax").plot(
                x,
                y,
                markers.get("marker") + markers.get("color"),
                markersize=markers.get("size"),
                label="Imported " + obj_type,
            )

    return fig_dict


def draw_mask(
    fig_dict: dict,
    grid,
    mask_to_plot: str,
) -> dict:

    xx = grid.longrid(native=True)
    yy = grid.latgrid(native=True)
    mask = grid.get(f"{mask_to_plot}_mask")
    if mask is None:
        return fig_dict

    mask = mask.astype(float)
    mask[mask < 1] = np.nan
    fig_dict.get("ax").contourf(xx, yy, mask, cmap="gray")

    return fig_dict


def draw_arrows(
    fig_dict: dict,
    lon: np.ndarray,
    lat: np.ndarray,
    xdata: np.ndarray,
    ydata: np.ndarray,
    scale: float = 100.0,
    reduce_arrows: Optional[int] = None,
):
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


def draw_graph_spectra1d(fig_dict, spec, freq, dirm, spr) -> dict:
    fig = fig_dict.get("fig")
    ax = fig_dict.get("ax")
    ax2 = fig_dict.get("ax2")
    ax.plot(freq, spec, color="blue", label="Spec ($m^2s$)", linewidth=2.5)
    if dirm is not None:
        ax2.plot(freq, dirm, color="g", label="dirm (deg)")
        if spr is not None:
            ax2.plot(freq, dirm - spr, color="red", ls="dashed", label="Spr (deg)")
            ax2.plot(freq, dirm + spr, color="red", ls="dashed")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    return fig_dict
