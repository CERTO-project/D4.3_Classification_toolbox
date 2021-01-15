"""
spectral library generation
"""

import os
import xarray as xr
from blob import glob
import argparse

def sample_file(file, step):
    """Take single netcdf and return regularly sampled pixels in a 2D array of dimension [wavelengths, pixels]"""

    ds = xr.open_dataset(file)
    ds = ds[[y for y in ds.data_vars if "Rrs" in y]].isel(
        latitude=slice(0,-1,step),
        longitude=slice(0,-1,step)
    )
    ds = ds.rename({v:int(v[-3:]) for v in ds.data_vars})
    ds = ds[[x for x in sorted(ds.data_vars)]]
    arr = ds.to_array(dim='wavelength')

    pixels = xr.DataArray(
        arr.data.reshape((11,-1)),
        dims=['wavelength','pixel'],
        coords={'wavelength':wavelengths}
    )

    # put negative values to zero?
    pixels = pixels.dropna(dim='pixel',how='any')

    return pixels.T.compute()

def main():



if __name__ == '__main__':
    main()
