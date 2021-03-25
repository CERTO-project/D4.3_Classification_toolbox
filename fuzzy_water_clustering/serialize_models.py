"""
Utilities for storing fitted model parameters.

AUTHOR: Angus Laurenson
EMAIL: anla@pml.ac.uk

Example:
    The attributions of a trained scikit-learn pipeline
    needs to be saved into netcdf:

        import xarray as xr

        ds_pl = pipeline_to_xarray(pipeline)
        ds_pl.to_netcdf(filename)

This is important metadata to include in netcdf files
of classified water spectra. It can be added to a
classified dataset like this:

        ds_out = ds_classified.merge(ds_pl)
        ds_out.to_netcdf(filename)

Todo:
    * Not sure.

"""

import sklearn
import xarray as xr
import numpy as np

def pipeline_to_xarray(pl):
    """ Takes a scikit-learn pipeline and
    returns an xr.Dataset containing parameters.
    
    Utility to be used for storing fitted model
    parameters to be included with classified data.

    NOTE:
    xarray requires arrays have named dimensions.
    Names are generated in order of appearance.
    Users must follow the transforms data to interpret
    their meaning. """


    # dicts, one for arrays and one for single values
    attrs_dict = {
        "pipeline":str(pl),
        'sklearn.__version__':sklearn.__version__
    }

    array_dict = {}

    # counter for dimension dummy names
    d = 0

    # for each step in the pipeline
    for step_name, step  in dict(pl.steps).items():

        # combine the step name with it's attribute names to flatten the dict tree
        for key, value in step.__dict__.items():

            # if an array, assign to array_dict for conversion to a xr.DataArray
            if type(value) == np.ndarray:
                array_dict.update({f"{step_name}.{key}":{"dims":[f"dummy_dim_{d+i}" for i in range(value.ndim)], "data":value}})
                d += value.ndim

            # else, assign to attrs_dict for conversion to xr.Dataset attributes
            else:
                attrs_dict.update({f"{step_name}.{key}":value})

    # create an xr.Dataset from the dictionary of arrays
    ds = xr.Dataset.from_dict(array_dict)

    # add the dictionary of single values as attributes
    ds.attrs = attrs_dict

    return ds
