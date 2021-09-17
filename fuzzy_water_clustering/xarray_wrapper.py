"""
This file contains functions to apply trained models to new data and package them up with the model metadata.

Author : Angus Laurenson, Plymouth Marine Laboratory
Email : anla@pml.ac.uk
"""

import xarray as xr
import dask.array as da
import dask.dataframe as dd
import numpy as np

class XarrayWrapper():
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
        wrapped = XarrayWrapper(estimator, **kwargs)

        # open netcdf with xarray
        ds = xr.open_dataset(netcdf_name)

        # fit model
        wrapped.fit(ds)

    """

    def __init__(self, model, **kwargs):
        # create the model
        self.model = model(**kwargs)


    def flatten_data(self, dataset):
        # reshape the dataset for scikit-learn (n_observations, m_features)
        # use unlabelled arrays to avoid overhead

        X = dataset.to_array(dim='variables')
        data = X.transpose(...,'variables').data.reshape(
            -1,
            X['variables'].size
        )

        return data

    def get_out_dimsize(self, X):
        # get the length of the new dimension
        # by running the estimator on a small sample

        # shorten data to 100 points (or less)
        try:
            X_small = X[:100,:]
        except:
            X_small = X

        Y_small = self.model.predict(X_small)

        return np.atleast_2d(Y_small).shape[0]

    def fit(self, dataset, Y=None):
        # takes xarray dataset, reshapes
        # fits model

        # record the training data variables and version
        self.data_vars_ = dataset.data_vars

        # reshape the dataset for scikit-learn (n_observations, m_features)
        data = self.flatten_data(dataset)

        # for training, drop nans
        X_train = dd.from_array(data).dropna().compute()

        # fit the model
        self.model.fit(X_train)

        # get the n_features_out
        self.n_features_out = self.get_out_dimsize(X_train)

    def predict_chunk(self, x, **kwargs):#method='default', chi2_metric='mahalanobis'):
        # reshapes a single chunk, applies model and re-reshapes back
        # assumes last dim is the feature dim
        chunk_shape = x.shape

        # flatten the chunk
        x = x.reshape(-1,chunk_shape[-1])

        # apply model
        y = self.model.predict(x, **kwargs).T#method=method, chi2_metric=chi2_metric).T

        # re-reshape chunk
        return y.reshape(chunk_shape[:-1]+(self.n_features_out,))

    def predict(self, dataset, **kwargs) -> xr.Dataset:#method='default', chi2_metric='mahalanobis'):
        # a parallelized predict step with options
        # @anla : this method does too much, maybe break it up

        # try to select vars
        assert set(self.data_vars_).issubset(set(dataset.data_vars)), \
                   f"given dataset variables {set(dataset.data_vars)} do not contain all expected variables {set(self.data_vars_)}"

        # does this preseve the order?
        X = dataset[list(self.data_vars_)]

        # get dimension order from variable
        new_dims = X[list(X.data_vars)[0]].dims

        # # for prediction, fill nans but save a mask
        mask = X[list(X.data_vars)[0]].isnull()
        
        # number of output features
        C = self.n_features_out
        
        # apply the model to chunkwise

        X = dataset.to_array(dim='variables')

        # @anla : bad naming here
        # set the variables to last position
        data = X.transpose(...,'variables').fillna(0)

        # set variable dim to have a single chunk
        data = data.chunk({'variables':-1})

        # take out the unlabelled array, bad naming
        data = data.data

        # @anla : it is annoying that the kwargs argument
        # does work in map_blocks
        membership_data = data.map_blocks(
            # lambda x:self.predict_chunk(x, method=method, chi2_metric=chi2_metric),
            lambda x:self.predict_chunk(x, **kwargs),
            chunks = data.chunks[:-1] + (tuple(C for x in data.chunks[-1]),),
            # kwargs = {'method':method,'chi2_metric':chi2_metric},
            dtype=float,
        ).persist()

        # print(membership_data.shape, membership_data.chunks)

        # copy the input dataset, dropping all variables
        ds_out = dataset.drop_vars(dataset.data_vars)

        # reshape the flattened memberhip array and put
        # into a clustered variable of the output dataset
        # reapply the mask of land and cloud
        # FIXME: is np.atleast_3d causing problems for 2D data?
        ds_out['clustered'] = xr.Variable(
            dims = list(new_dims) + ['optical_water_type'],
            data = membership_data,
            attrs= {}
        ).where(mask==0)

        # split the clustered variable into seperate data variables
        ds_out['optical_water_type'] = [f'owt_{x}' for x in range(C)]
        ds_out = ds_out['clustered'].to_dataset(dim='optical_water_type')

        return ds_out
