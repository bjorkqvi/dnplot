import xarray as xr
import pandas as pd
def force_to_ds(ds):
    """Takes a dict, dnora ModelRun and gets the 'waveseries' object's xr.Dataset.
    If a geo-skeleton is given, then that dataset is returned.
    If a dataset is given, it is returned."""
    if not ds:
        return None
    ds = ds.get('waveseries') or ds
    if not isinstance(ds,xr.Dataset):
        ds = ds.ds()
    return ds

def get_units(ds, var:str) -> str:
    """Takes the units from a xr.Dataset"""
    return getattr(ds.get(var), 'units', '')

def get_varname(ds, var: str) -> str:
    """Gets a long_name, standard_name or short_name from a dataset"""
    return (
        getattr(ds.get(var), 'long_name', None) or
        getattr(ds.get(var), 'standard_name', None) or
        getattr(ds.get(var), 'short_name', var)
    )

def xarray_to_dataframe(ds) -> pd.DataFrame:
   
    df = ds.to_dataframe()
    df = df.reset_index()
    col_drop = ["lon", "lat", "inds"]
    df = df.drop(col_drop, axis="columns")
    df.set_index("time", inplace=True)
    df = df.resample("h").asfreq()
    df = df.reset_index()
    return df
