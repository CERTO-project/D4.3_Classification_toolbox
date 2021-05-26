"""
This file contains functions to apply trained models to new data and package them up with the model metadata.

Author : Angus Laurenson, Plymouth Marine Laboratory
Email : anla@pml.ac.uk
"""

import xarray as xr
import dask.array as da
import dask.dataframe as dd
import numpy as np

class NetcdfWrapper():
    """
    This wrapper is intended for scikit-learn estimators
    to enable them to be applied to xarray.datasets.

    It changes their expected input, X, from a 2d array
    of (n observations, m features), to an xarray.dataset
    that has several data variables that share several
    dimensions. Dataset variables are considered features
    whilst dataset dimensions and coordinates are discarded.

    In addition, NetcdfWrapper() stores the variables and
    processing_level of the netcdf data on which it was fitted

    USAGE :
        # create model
        model = NetcdfWrapper(estimator, **kwargs)

        # open netcdf with xarray
        ds = xr.open_dataset(netcdf_name)

        # fit model
        model.fit(ds)

    """

    def __init__(self, model, **kwargs):
        # create the model
        self.model = model(**kwargs)


    def flatten_data(self, dataset):
        # reshape the dataset for scikit-learn (n_observations, m_features)
        # use unlabelled arrays to avoid overhead

        X = dataset.to_array(dim='variables')
        data = X.transpose(...,'variables').data.reshape(-1,X['variables'].size)

        return data

    def fit(self, dataset, Y=None):
        # takes xarray dataset, reshapes
        # fits model

        # record the training data variables and version
        self.data_vars_ = dataset.data_vars
        self.processing_level_ = dataset.processing_level

        # reshape the dataset for scikit-learn (n_observations, m_features)
        data = self.flatten_data(dataset)

        # for training, drop nans
        X_train = dd.from_array(data).dropna().compute()

        # fit the model
        self.model.fit(X_train)

    def predict(self, dataset):
        # a parallelized predict step

        # try to select vars
        assert set(self.data_vars_).issubset(set(dataset.data_vars)), \
                   f"given dataset variables {set(dataset.data_vars)} do not contain all expected variables {set(self.data_vars_)}"

        # does this preseve the order?
        X = dataset[list(self.data_vars_)]

        assert self.processing_level_ == X.attrs['processing_level'], \
                   f"given dataset processing level \"{X.attrs['processing_level']}\" does not match expected processing level \"{self.processing_level_}\""

        # reshape the dataset for scikit-learn (n_observations, m_features)
        data = self.flatten_data(X.fillna(0))

        # for prediction, fill nans but save a mask
        mask = X[list(X.data_vars)[0]].isnull()
        data = np.nan_to_num(data)
        data = data.rechunk(('auto',-1))

        # print(f"shape = {data.shape}, chunks = {data.chunks}")

        # number of output features
        # FIXME: specific to CmeansModel right now...
        C = self.model.c
        M = X[list(X.data_vars)[0]].size

        # print(f"C = {C}, M = {M}")

        # apply the model to chunkwise
        membership_flattened = data.map_blocks(
            lambda x : self.model.predict(x),
            chunks=(data.chunks[0],(C)),
            dtype=float,
        ).persist()

        # copy the input dataset, dropping all variables
        ds_out = dataset.drop_vars(dataset.data_vars)

        # print(f"membership flattened = {membership_flattened}")

        # reshape the flattened memberhip array and put
        # into a clustered variable of the output dataset
        # reapply the mask of land and cloud
        ds_out['clustered'] = xr.Variable(
            dims = list(ds_out.dims)+['optical_water_type'],
            data = da.atleast_3d(membership_flattened.T.reshape([ds_out.dims[x] for x in ds_out.dims] + [C])),
            attrs= {}
        ).where(mask==0)

        # split the clustered variable into seperate data variables
        ds_out['optical_water_type'] = [f'owt_{x}' for x in range(C)]
        ds_out = ds_out['clustered'].to_dataset(dim='optical_water_type')

        return ds_out
