"""
spectral library generation
"""

import os
import xarray as xr
from glob import glob
import argparse

def sample_file(
    file,
    step_size=100,
    variables=lambda x : "Rrs" in x,
    mask=None,
    sensor=None,
    time_bounds=None,
    max_pixels=None,
):
    """Given a filename,
    return a 2D array with dimensions (bands, pixels)

    Used for generating training data for water spectra
    analysis models

    Arguments:

     - files : a list of files to sample from. Choose carefully
     - list or function to filter the file's data_vars
     - step_size : data indexed in lat and lon as [::step_size]
     - mask : for excluding regions. Must match grid
     - sensor : if file list is mixed, choose sensor (required?)
     - time_bounds : (tmin, tmax) or pandas datetime index (required?)
     - max_pixels : limit on the number of pixels returned

    Returns:
     - xarray.DataArray with dimensions (bands, pixels) and a size
     that is determined by the number of files and step_size or limited
     to max_pixels if defined.

    """

    ds = xr.open_dataset(file)

    # if the variables aren't a list, try filtering
    if type(variables) != list:
        variables = list(
            filter(
                variables,
                ds.data_vars
            )
        )

    # crop down to the variables and subset data
    ds = ds[variables].isel(
        latitude=slice(0,-1,step_size),
        longitude=slice(0,-1,step_size)
    )

    # if mask is supplied. apply mask.

    if type(mask) == str:
        mask = xr.open_dataarray(mask)

    if type(mask) == xr.DataArray:

        try:
            ds = ds.where(mask == True)

        except:
            print("Invalid mask, skipping mask step")


    # try to sort by wavelength
    try:
        ds = ds.rename({v:int(v[-3:]) for v in ds.data_vars})
        ds = ds[[x for x in sorted(ds.data_vars)]]
        dname='wavelength'

    # if cannot convert to ints and order, then raise warning
    except:
        dname='variables'

    arr = ds.to_array(dim=dname)

    pixels = xr.DataArray(
        arr.data.reshape((11,-1)),
        dims=[dname,'pixel'],
        coords={dname:arr.coords[dname]}
    )

    # drop null values (they're just bits of the mask)
    pixels = pixels.dropna(dim='pixel',how='any')

    return pixels.T.compute()
