"""

The idea is to run a full workflow with a tiny and awkward dataset, testing the normal operation of every piece of the project.

utils.sample_file()

estimators.CmeansModel()

utils.apply_model()

"""
import sys
sys.path.append("../")

import fuzzy_water_clustering as fwc

import xarray as xr

import skimage
import os

# # # # CREATE TEST DATA USING SKIMAGE DATASETS # # # # #

if os.path.isfile('./coffee.nc') == False:

    img = skimage.data.coffee()

    da = xr.DataArray(
        data = img,
        dims = ('x','lat','vars'),
        coords = {
            'x':range(img.shape[0]),
            'lat':range(img.shape[1]),
            'vars':['1vand_1','Rrs_422','test']
            }
    )

    ds = da.to_dataset(dim='vars').to_netcdf('coffee.nc')

# # # # # SAMPLE FILE TEST # # # # #
training_data = fwc.sample_file('coffee.nc', step_size=10, variables='none')

# make an assertion about the shape of the sampled data

# # # # # CLUSTERING ESTIMATOR TEST # # # # #
cmeans = fwc.CmeansModel(c=3,m=1.5)
cmeans.fit(training_data)

# # # # # APPLY TRAINED MODEL USING PREDICT FILE # # # # #
ds_classified = fwc.predict_file('coffee.nc', cmeans, variables='all', store_model=False)
