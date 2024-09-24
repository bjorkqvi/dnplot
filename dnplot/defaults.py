import cmocean.cm


default_variable = {
    "hs": {"name": "Significant wave height", "unit": "m", "cmap": cmocean.cm.amp},
    "wind": {"name": "Wind", "unit": "m/s", "cmap": cmocean.cm.tempo},
    "current": {"name": "Current", "unit": "m/s", "cmap": cmocean.cm.tempo},
    "topo": {
        "name": "Topography",
        "unit": "m",
        "cmap": cmocean.tools.crop_by_percent(cmocean.cm.topo_r, 50, which="min"),
    },
    "mask": {"name": " ", "unit": " ", "cmap": "gray"},
}

default_markers = {
    # To have something to plot for all objects
    "generic_objects": {"marker": "x", "color": "m", "size": 2},
    "generic_points": {"marker": "*", "color": "m", "size": 2},
    # These are objects that represent DNORA unstructured data
    "spectra": {"marker": "x", "color": "k", "size": 7},
    "waveseries": {"marker": "x", "color": "r", "size": 7},
    # These are objects that represent DNORA gridded data
    "wind": {"marker": ".", "color": "k", "size": 1},
    "current": {"marker": ".", "color": "r", "size": 1},
    "ice": {"marker": ".", "color": "b", "size": 1},
    # These are points that correspond to boolean masks
    "boundary_points": {"marker": "*", "color": "k", "size": 5},
    "output_points": {"marker": "*", "color": "r", "size": 5},
}
