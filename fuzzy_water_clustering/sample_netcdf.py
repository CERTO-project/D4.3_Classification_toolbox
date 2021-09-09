"""
Sampling functions for Xarray.DataArray objects

Used for sampling training data from larger netcdf files

AUTHOR
    Angus Laurenson
    Plymouth Marine Laboratory
    anla@pml.ac.uk

"""

import xarray as xr
import re

def random_coarsener(x, axis, **kwargs):
    """collapse an array along the given axis,
    use with xr.Dataset.coarsen() to randomly
    keep 1 pixel from each window
    
    Provides a even coverage yet still random,
    a compromise between random and stepping
    that avoids row hammering.
    """
    
    # for each axis of the array,
    # randomly select an index value or not depending
    # on whether its included in the axis arguement
    
    indices = []
    for i in range(x.ndim):
        if i in list(axis):
            indices.append(choice(np.arange(x.shape[i])))
        else:
            indices.append(slice(None,None,None))
    
    # indexing works different for tuples and lists...
    return x[tuple(indices)]
    

def stack_dataarray_drop_index(da:xr.DataArray, feature_dim:, **kwargs) -> xr.DataArray:
    """Destroy all other dims except the given feature_dim,
    return a 2D xr.DataArray with shape (observations, features)"""
    
    # put feature dim last, as per sklearn
    da = da.transpose(...,feature_dim)
    
    # reshape unlabelled array
    arr = da.data.reshape(-1,da[feature_dim].size)

    # create a new data array of flattened values
    da_out = xr.DataArray(
        data=arr,
        dims=('pixel',feature_dim),
        coords={feature_dim:da[feature_dim]}
        attrs = da.attrs
    )
    
    # remove any pixel that contains a nan
    da_out = da_out.dropna(dim='pixel')
    
    return da_out
        

def stack_dataset_drop_index(ds:xr.Dataset, **kwargs) -> xr.Dataset:
    """take k random samples from a dataset,
    assumes that data variables are the features,
    flattens all coordinates and selects sample"""

    # convert to dataarray
    da = ds.to_array(dim='feature')
    
    # reuse function for dataarrays
    da_stacked = stack_dataarray_drop_index(
        da,
        feature_dim='feature',
        **kwargs
    )
    
    # put back into dataset and return
    return da_stacked.to_dataset(dim='variable')

def random_sample_array(da:xr.DataArray, k=100, **kwargs):
    """given a 2D array (xr.DataArray or dask.array),
    it must have axes ordered (observations, features),
    pick a random sample along the observations"""
    
    indices = np.arange(da.shape[0])
    
    random_indices = sorted(choices(indices, k=k))
    
    return da[random_indices]



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
     - step_size : a dict or iterable of ints, sample every nth pixel
     - mask : for excluding regions. Must match grid
     - sensor : if file list is mixed, choose sensor (required?)
     - max_pixels : limit on the number of pixels returned

    Returns:
     - xarray.DataArray with dimensions (bands, pixels) and a size
     that is determined by the number of files and step_size or limited
     to max_pixels if defined.

    """


    # Accept different input types, str, dataset or

    if type(file) == str:
        ds = xr.open_dataset(file)

    else:
        ds = file


    # accept different variable arugment types

    if type(variables).__name__ == "function":
        variables = list(
            filter(
                variables,
                ds.data_vars
            )
        )

    if type(variables) == list:
        ds = ds[variables]

    # apply the step_size variable

    # accept dict of (dim:int) pairs
    if type(step_size) == dict:
        ds = ds.isel(
            **{k: slice(None,None,v) for (k,v) in step_size.items()}
        )

    # accept list of ints, apply as slice as if unlabelled
    elif (type(step_size) == tuple) | (type(step_size) == list):
        ds = ds.isel(
            **{var:slice(None,None,x) for (var,x) in zip(ds.dims,step_size)}
        )

    elif type(step_size) == int:
        ds = ds.isel(
            **{dim: slice(None,None,step_size) for dim in ds.dims}
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
        ds = ds.rename({v:int(re.findall(r'\d+', v)[0]) for v in ds.data_vars})
        ds = ds[[x for x in sorted(ds.data_vars)]]
        dname='wavelength'

    # if cannot convert to ints and order, then raise warning
    except:
        dname='variables'

    arr = ds.to_array(dim=dname)
    
    if return_locations == True:
        ds.stack(pixel=('x','y'))

    pixels = xr.DataArray(
        arr.data.reshape((arr[dname].size,-1)),
        dims=[dname,'pixel'],
        coords={dname:arr.coords[dname]}
    )

    # drop null values (they're just bits of the mask)
    pixels = pixels.dropna(dim='pixel',how='any')

    return pixels.T.compute()
