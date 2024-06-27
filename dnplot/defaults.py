import cmocean.cm


default_variable = {
    "hs": {"name": "Significant wave height", "unit": "m", "cmap": cmocean.cm.amp},
    "ff": {"name": "Wind", "unit": "m/s", "cmap": cmocean.cm.tempo},
    "topo": {
        "name": "Topography",
        "unit": "m",
        "cmap": cmocean.tools.crop_by_percent(cmocean.cm.topo_r, 50, which="min"),
    },
    "mask": {"name": " ", "unit": " ", "cmap": "gray"},
}

default_markers = {
    "generic_objects": {"marker": "x", "color": "m", "size": 2},
    "generic_masks": {"marker": "*", "color": "m", "size": 2},
    "spectra": {"marker": "x", "color": "k", "size": 5},
    "wind": {"marker": ".", "color": "r", "size": 1},
    "spectra1d": {"marker": "o", "color": "k", "size": 3},
    "spectra_mask": {"marker": "*", "color": "r", "size": 7},
    "output_mask": {"marker": "d", "color": "b", "size": 5},
}
