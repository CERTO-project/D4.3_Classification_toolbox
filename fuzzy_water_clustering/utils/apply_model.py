"""
This file contains functions to apply trained models to new data and package them up with the model metadata.

Author : Angus Laurenson, Plymouth Marine Laboratory
Email : anla@pml.ac.uk
"""

import numpy as np
import xarray as xr
import dask.array as da
from .serialize_models import pipeline_to_xarray

def choose_power(L):
    d = 1
    n=1
    while L % d == 0:
        n += 1
        d = 2**n
    return 2**(n-1)

def predict_file(dataset, model, variables=lambda x:("Rrs_" in x)&(len(x) == 7), store_model=True, **predict_kwargs):
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

    # if the dataset argument is a string, assume it is a filename
    if type(dataset) == str:
        dataset = xr.open_dataset(dataset, chunks={})

    # if the dataset isn't an xr.Dataset raise error
    if type(dataset) != xr.Dataset:
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

    # filter the data variables by the variables function
    if str(type(variables)) == "<class 'function'>":
        variables = list(filter(variables, dataset.data_vars))

    # select only the variables given or filtered
    ds = dataset[variables]

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
        warnings.warn(
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
    data = ds.fillna(0).data.reshape((n_features,-1)).T

    # use dask array for parallelization
    chunk_size = choose_power(data.shape[0])

    data = data.rechunk((chunk_size,n_features))

    # apply the model to chunkwise
    mem = data.map_blocks(
        lambda x : model.predict(x, **predict_kwargs),
        chunks=(data.chunks[0],C),
        dtype=float,
    ).compute()

    print(f"mem.shape {mem.shape}, chunk_size {chunk_size}, data.shape {data.shape}")

    # turn the classified data into a xr.Dataset
    # chopping and reshaping the data is required
    # FIXME (anla) : what if time dimension, what if names not latitude..
    classified = xr.DataArray(
        data = np.hstack(
            np.vsplit(
                mem, data.shape[0] // chunk_size
                )
            ).reshape(
                [C] + [ds[x].size for x in ds.coords if x != dname]
            ),
        dims=(['owt'] + [x for x in ds.dims if x != dname]),

        # this seems like a goofy way to copy coords but OK!
        # ValueError: coordinate wavelength has dimensions ('wavelength',), but these are not a subset of the DataArray dimensions ['owt', 'time', 'latitude', 'longitude']
        coords = {**{'owt':range(C)},
        **{k:v.values for (k,v) in dict(ds.coords).items() if k != dname}}
    )
    #removed a sortby('latitude') might need a flip?

    # apply mask from original data to classified data
    # also adds time coords + wavelength. Drop the latter
    classified = classified.where(mask)


    if store_model == True:
        # serialize model parameters to add to file?
        ds_model = pipeline_to_xarray(model)

        # merge the dataset of classified data with the one for model parameters
        classified = xr.merge((classified.to_dataset(name='classified_data'),ds_model))


    return classified #.sortby('latitude')
