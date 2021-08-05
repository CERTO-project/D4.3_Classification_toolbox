"""
This file contains functions to apply trained models to new data and package them up with the model metadata.

Author : Angus Laurenson, Plymouth Marine Laboratory
Email : anla@pml.ac.uk
"""

import numpy as np
import xarray as xr
import dask.array as da
from .serialize_models import pipeline_to_xarray
import logging

logger = logging.getLogger()

stream_handler = logging.StreamHandler()

logger.addHandler(stream_handler)

def choose_power(L):
    d = 1
    n = 0
    while L % d == 0:
        n += 1
        d = 2**n
    return 2**(n-1)

def predict_file(file, model, variables=lambda x:("Rrs_" in x)&(len(x) == 7), store_model=True, **predict_kwargs):
    """Given a netcdf file name or xr.Dataset and a fitted model.
    Return an xr.Dataset that contains
    the classified data and model parameters.

    Operation:
        model.predict() is applied blockwise to the dataset
        the original datasets' mask and coordinates are preserved
        model parameters are added to the output

        if variables are not declared it looks for Rrs_ variables
        and will order them by wavelength. A warning is raised
        otherwise.

    Usage:
        ds_classified = predict_file(file_name, pipeline)
    """

    logger.warning("Depreciation warning: predict_file is being replaced by XarrayWrapper")

    # # # HANDLE DATA INPUT # # #

    # if the dataset argument is a string, assume it is a filename
    if type(file) == str:
        ds = xr.open_dataset(file, chunks={})

    else:
        ds = file

    # if the dataset isn't an xr.Dataset raise error
    if type(ds) != xr.Dataset:
        raise Exception(
            "Invalid 'dataset' argument, use\n\
            either filename or xarray.Dataset object"
        )

    # get the number of clusters
    try:
        # for single estimator
        """WARNING what if the number of clusters isn't called 'c' ?"""
        C = model.get_params()['c']
    except:
        # for pipelines, clustering should come last?
        C = model.steps[-1][-1].get_params()['c']

    if type(variables).__name__ == "function":
        variables = list(
            filter(
                variables,
                ds.data_vars
            )
        )

    if type(variables) == list:
        ds = ds[variables]

    else:
        variables = list(ds.data_vars)


    # copy mask from original dataset to be added at end.
    # Required as .predict() will fail if nans are present.
    mask = xr.where(ds[variables[0]].isnull(), False, True)

    # try to sort by wavelength
    try:
        ds = ds.rename({v:int(v[-3:]) for v in ds.data_vars})
        ds = ds[[x for x in sorted(ds.data_vars)]]

        dname='wavelength'

        ds = ds.to_array(dim=dname)
        ds['wavelength'].attrs["units"] = 'nm'

    # if cannot convert to ints and order, then raise warning?
    except:
        dname='variables'
        logger.warning(
            "Expected variable names ending with 3 digits\n\
            indicating integer wavelengths in nanometers\n\
            not found. Therefore proceeding with,\n\
            dim = 'variables' instead.")

        ds = ds.to_array(dim=dname)

    # single chunk over features requried for reshape
    ds = ds.chunk({dname:-1,})
    n_features = ds[dname].size

    # fillna() required for predict step, mask re-added after
    # reshape required to obtain 2D array for scikit-learn
    # use unlabelled, dask arrays, for faster reshaping
    data = ds.fillna(0).data.reshape((n_features,-1))

    # apply the model to chunkwise
    mem = data.map_blocks(
        lambda x : model.predict(x.T).T,
        chunks=((C),data.chunks[1][::]),
        dtype=float,
    ).compute()

    classified = ds[:C,].copy()

    classified.data = mem.reshape((C,*ds.shape[1:]))

    # classified = classified.where(ds[:C,].isnull() == False)

    # FIXME: hardcode!
    # classified = classified.sortby('latitude')

    # name the optical water type dimension "owt"
    classified.rename({dname,'owt'})

    # apply mask from original data to classified data
    # also adds time coords + wavelength. Drop the latter
    classified = classified.where(mask)

    # optionally store model parameters inside the netcdf
    if store_model == True:
        # serialize model parameters to add to file?
        ds_model = pipeline_to_xarray(model)

        # merge the dataset of classified data with the one for model parameters
        classified = xr.merge((classified.to_dataset(name='classified_data'),ds_model))


    return classified.to_dataset(name='classified_data')
