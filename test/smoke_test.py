"""
SMOKE TEST of fuzzy-water package

AUTHOR:
    Angus Laurenson, Plymouth Marine Laboratory
    anla@pml.ac.uk

DESCRIPTION
    A simulation of the typical workflow:
    1. sampling data
    2. clustering sample
    3. apply cluster set to datasets

"""
import sys
sys.path.append("../")

import fuzzy_water_clustering as fwc

import xarray as xr

# import and setup logging
import logging
logger = logging.getLogger(name="smoke_test_logger")
logstream = logging.StreamHandler()
logger.addHandler(logstream)

import os

# # # # CREATE TEST DATA USING SKIMAGE DATASETS # # # # #

# load a dataset from the tutorial
test_ds = xr.tutorial.open_dataset('eraint_uvz', chunks='auto')

# write to file
test_ds.to_netcdf(
    'tutorial.eraint_uvz.nc',
    encoding={x:{'zlib':True,'complevel':4} for x in test_ds.data_vars}
)

# sample file
training_data = fwc.sample_file('coffee.nc', step_size=10, variables='none')

# create CmeansModel instance and fit
cmeans = fwc.CmeansModel(c=3,m=1.5)
cmeans.fit(training_data)

# apply model to file using predict_file
ds_classified = fwc.predict_file('coffee.nc', cmeans, variables='all', store_model=False)

# write to file
ds_classified.to_netcdf(
    "clustered_eraint_uvz.nc",
    encoding={x:{'zlib':True,'complevel':4} for x in ds_classified.data_vars}
)

# test the XarrayWrapper
wrapped = fwc.XarrayWrapper(fwc.CmeansModel, C=5)

# fit to xarray.Dataset
wrapped.fit(test_ds)

# predict xarray.Dataset
test_pred_chi2 = wrapped.predict(test_ds, method='chi2')

# write to netcdf
test_pred_chi2.to_netcdf(
    "test_chi2.nc",
    encoding={x:{'zlib':True,'complevel':4} for x in test_ds.data_vars}
)