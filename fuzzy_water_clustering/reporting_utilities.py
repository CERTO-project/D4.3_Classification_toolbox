"""
Utilities for plotting fuzzy water clustering results

AUTHOR:
    Angus Laurenson
    Plymouth Marine Laboratory
    anla@pml.ac.uk

DESCRIPTION:
    A submodule of fuzzy-water package that contains functions
    for creating holoviews/bokeh plots showing the results of
    clustering.
    
    Includes functions for:
    
        plotting the scores from GridSearchCV, varing C and M
    
        plotting the mean and standard deviation of clusters in the
        original features space, assumed to be:

            x = wavelength (nm)
            y = RRS (ratio)
        
        geographic plot of cluster memberhips and dominant cluster
    
"""

import xarray as xr
import hvplot.xarray
import fuzzy_water_clustering as fwc
import pandas as pd
import hvplot.pandas
import numpy as np
from functools import reduce
from random import choices
from datetime import datetime

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


def prepare_dataset(ds):
    """Correct mistakes in S2 procesed data for REVIVAL"""
    
    # assign lat lon as coords not variables
    ds = ds.assign_coords(coords={
        'lon':ds['lon'],
        'lat':ds['lat']
    })

    # select RRS variables (bands) only
    ds = ds[[x for x in ds.data_vars if x.startswith("Rrs_")]]

    ds = ds.assign_coords(coords={'time':datetime.strptime(ds.attrs['isodate'][:10], "%Y-%m-%d")})
    ds = ds.expand_dims('time')
    
    ds.attrs['units'] = 'ratio'

    for var in ds.data_vars:
        ds[var].attrs['units'] = 'ratio'
        
    return ds

def get_sample(all_files, sample_size=10, **kwargs):
    """take random sample of sample_size from list of all_files,
    evenly subsample with a given step_size and concatenate the
    results into a single dataset for training"""
    
    # take a sample of files without replacement
    sample_files = choices(all_files, k=sample_size)
    
    # open the files, correct the mistakes if needed
    try:
        ds_list = [fwc.sample_file(f, **kwargs) for f in sample_files]
    except:
        ds_list = [
            fwc.sample_file(
                prepare_dataset(
                    xr.open_dataset(file, decode_cf=False, mask_and_scale=True)
                ),
                **kwargs
            ) for file in sample_files
        ]
        
    return xr.combine_nested(ds_list, 'pixel')

def plot_scores(df,model,scoring):
    return \
    reduce(lambda a,b:a*b,[df.hvplot.errorbars(
            col='param_cmeansmodel__m',
            row=None,
            x='param_cmeansmodel__c',
            y=f'mean_test_{x}',
            yerr1=f'std_test_{x}',
        )*df.hvplot(
            col='param_cmeansmodel__m',
            row=None,
            kind='scatter',
            x='param_cmeansmodel__c',
            y=f'mean_test_{x}',
            label=f'{x}',
            legend=True,
#             ylim=[-1.5,1.5],
            ylabel="score",
            xlabel="# clusters c",
    ) for x in scoring.keys()]).opts(title=f"Pipeline = {list(model.named_steps.values())}", show_legend=True)

def invert_transformations(model:Pipeline, Y):
    """inverse_transform each transformation in a Pipeline
    to project an array back to the original feature space"""
    
    for step in model.steps[::-1]:
        
        transformer = step[1]
        
        if isinstance(transformer, TransformerMixin):
            Y = transformer.inverse_transform(Y)
    
    return Y
    
def get_standard_devation(data, memberships):
    """given training data and memberships
    calculate the weighted standard devation
    of the memberships to each cluster"""
    covs = []
    for mems in memberships:
        covs.append(np.cov(data.T,aweights = mems))
       
    # diagonal of covariance is just variance,
    # square root of variance is standard deviation
    # which has the same units as the variable..
    stds = [np.diag(cov)**0.5 for cov in covs]
    
    return np.stack(stds)

def plot_centers(model, ds_train):
    """a large function that takes a model and training data
    returns a holoviews plot of mean and standard devation"""    
    
    # transform back to original space
    centers = invert_transformations(
        model,
        model.named_steps['cmeansmodel'].cntr_
    )
    
    memberships = model.named_steps['cmeansmodel'].u_
    
    # compute standard deviation
    # in original space
    stds = get_standard_devation(ds_train, memberships)
    
    # make dataframes with correct labels,
    # useful for wrangling and accessing hvplot
    df_c = pd.DataFrame(
        centers.T,
        index=ds_train['wavelength']
    )
    
    df_std = pd.DataFrame(
        stds.T,
        index=ds_train['wavelength']
    )
    
    # create a list of plots, one for each cluster
    # to be put together later by calling reduce(...)
    std_plots = [
        pd.DataFrame(
            {
                'mean':df_c[col],
                'y':df_c[col]-df_std[col],
                'y2':df_c[col]+df_std[col]
            }
        ).hvplot.area(
            x='wavelength',
            y='y',
            y2='y2',
            alpha=0.2,
        ) for col in df_c.columns
    ]

    
    mean_std_plot = (reduce(lambda a,b:a*b, std_plots) * df_c.hvplot(group_label='Cluster')).opts(
        title=f"Cluster means: {list(model.named_steps.values())}",
        xlabel='wavelength (nm)',
        ylabel='RRS (ratio)',
        legend_opts={'title':'Cluster'}
    )
    
    return mean_std_plot

