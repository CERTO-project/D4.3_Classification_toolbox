"""
Description:
    A set of cluster scoring objects for evaluating unsupervised clustering algorithms and pipelines.

    Author - Angus Laurenson, Plymouth Marine Laboratory
    Email  - anla@pml.ac.uk

ToDo:
    # FIXME : WOULD A DECORATOR FUNCTION SUFFICE HERE? does it already exist in sklearn?
    # NOTE : we can probably get a lot of these indices also from R over rpy2, as long as we can feed in input in the form
"""

import sklearn
from sklearn.metrics import pairwise_distances, make_scorer
from sklearn.metrics.cluster import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
import numpy as np
from sklearn.utils.validation import check_array

def diff_to_default(estimator, x, y=None):
    """scoring function for fuzzy clustering
    
    returns:
     negative square root of the mean of the 
     squared difference between default and chi2 predictions
    
    Higher value is better, scores useful to choose 
    the degrees_freedom parameter for the chi2 method"""

    return -np.sqrt(
            np.mean((
                estimator.predict(x, method='default') - estimator.predict(x, method='chi2')
            )**2)
        )

def xie_beni(Estimator, X, y=None):
    """ Xie-Beni scoring function
        for cmeans.

        Output is negative as scikit-learn
        by default maximizes scores """

    u = Estimator.predict(X)
    v = Estimator["cmeansmodel"].cluster_centers_
    m = Estimator["cmeansmodel"].m

    n = X.shape[0]
    c = v.shape[0]

    um = u**m

    d2 = pairwise_distances(X, v)
    v2 = pairwise_distances(v, v)

    v2[v2 == 0.0] = np.inf

    return -np.sum(um.T*d2)/(n*np.min(v2))

def hard_silouette(Estimator, X, y=None):
    """ A hard silouette scoring function
        for clustering algorithms. Built on
        `sklearn.metrics.silhouette_score`

        Uses np.argmax() to convert soft
        clusters into hard clusters prior
        to evaluation."""

    u = Estimator.predict(X)
    if u.shape != X.shape[0]:
        u = np.argmax(u, axis=0)

    return silhouette_score(X, u)

def fuzzy_partition_coef(Estimator, X, y=None):
    """ Fuzzy partion coefficient (fpc)
        for fuzzy clustering algorithms"""

    u = Estimator.predict(X)

    if type(Estimator) == sklearn.pipeline.Pipeline:
        return Estimator['cmeansmodel'].fpc_
    else:
        return Estimator.fpc_

def calinski_harabasz(Estimator, X, y=None):
    """ Fuzzy partion coefficient (fpc)
        for fuzzy clustering algorithms"""

    u = Estimator.predict(X)

    if u.shape != X.shape[0]:
        u = np.argmax(u, axis=0)

    return calinski_harabasz_score(X, u)


def davies_bouldin(Estimator, X, y=None):
    """ Fuzzy partion coefficient (fpc)
        for fuzzy clustering algorithms"""

    u = Estimator.predict(X)

    if u.shape != X.shape[0]:
        u = np.argmax(u, axis=0)

    return -davies_bouldin_score(X, u)
