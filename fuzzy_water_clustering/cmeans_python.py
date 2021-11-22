'''
# CmeansModel
A scikit-learn compatible cmeans clustering estimator object.
Built on skfuzzy.cluster.cmeans and cmeans_predict functions.

`from fwc import CmeansModel`

See class docstring for more info.

Author : Angus Laurenson
Email  : anla@pml.ac.uk

FIXME @anla : there is feature envy CmeansModel replicates a lot
of the features/attributes froms cmeans() from scikit-fuzzy

'''

# cmeans algorithms
from numpy.lib.twodim_base import diagflat
from skfuzzy.cluster import cmeans, cmeans_predict

# for data handling
import numpy as np

# scikit-learn checks
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

# scikit-learn base classes
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.decomposition import PCA

# distances
from scipy.spatial.distance import cdist
from scipy.stats import chi2

def get_degrees_freedom(x, threshold=0.99)->int:
    """Get the number of principal components that explain
    up to the given threshold of the variance of a given dataset
    the default is (default 99%) and that is usually 3 or 4
    for ocean colour imagery"""
    pca = PCA()
    pca.fit(x)

    degrees_freedom = np.argwhere(
        np.cumsum(pca.explained_variance_ratio_)>=0.99
    )

    return int(degrees_freedom.flat[0])

def _chi2_predict(x, cluster_centers_, VI, degrees_freedom=-1, metric='euclidean') -> np.array:
    """Custom function that assigns unnormalised fuzzy membership 
    values according to the chi2.sf distribution function 
    
    args:

    x : 2d array of (n_observations, n_features)
    VI : 3d array of shape (n_clusters, n_features, n_features), 
        stacked inverse covariance matrices from each cluster

    kwargs:

    degrees_freedom : int
        Defaults to -1, which resorts to n_features
        number of degrees of freedom passed to chi2.sf as the df kwarg
        Best practice is to perform PCA on the training data and then
        set degrees_freedom to the number of components that caputres
        99% of the data. In ocean colour that number is usually 3 or 4.
        
    metric : str
        what metric space to measure distance in. Euclidean is default,
        mahalanobis can also be used"""

    # explicitly define number of features and classes
    n_classes, n_features = cluster_centers_.shape

    # check the data
    x = check_array(x)

    # assert the number of features matches
    assert x.shape[1] == n_features, "number of features inconsistent between clusters and x"

    # create an array of zeros to fill with memberships
    # shaped as x, but n_features is replaced with n_classes    
    memberships=np.zeros(x.shape[:-1]+(n_classes,))
    
    # for each class calc distance
    # FIXME @anla : incompatible arguments
    dist = cdist(
        x.reshape(-1,n_features),
        cluster_centers_,
        metric=metric,
        VI=VI
    )
    
    # default is to use the number of features
    if degrees_freedom == -1:
        degrees_freedom = n_features
    
    # this is post-haste, should be done on the training data
    if degrees_freedom == 'auto':
        degrees_freedom = get_degrees_freedom(x)

    # calculate the membership
    memberships=chi2.sf(dist**2, df=degrees_freedom).squeeze()
    
    return memberships.T


class CmeansModel(BaseEstimator, ClusterMixin):
    """A c-means clustering estimator that is compatible with scikit-learn.

    Built on skfuzzy.cluster.cmeans and cmeans_predict functions.
    fit() and predict() methods have original docstrings

    # USAGE #

    cmeans = CmeansModel(
        n_clusters=5,
        m=2,
        tol=0.005,
        maxiter=1000,
        random_state=None,
        random_starts=1,
        distance_metric='euclidean'
    )

    # determine clusters
    cmeans.fit(x)

    # estimate memberships
    lables = cmeans.predict(x)

    # score clustering
    score = cmeans.score()

    # INIT ARGUMENTS #

    n_clusters is int number of clusters
    m is fuzziness factor ~ 1.5 < M < 2.5
    tol is the error tolerance of the iteration
    maxiter is maximum number of iteration
    random_state is for scikit-learn learn not sure
    scoring_metric is the name of method to use when scoring, used for finding optimum parameters

    # METHODS #

    fit(x) -> self
    predict(x) -> array
    fit_predict(x) -> array
    score() -> float

    where x is an array of size (N,M),
    N is the number of observations,
    M is the number of of features.

    # ATTRIBUTES #
    * fitted parameters are indicated with trailing underscores
    
    cluster_centers_ = fitted cluster centers
    u_ = array of cluster memberships
    labels_ = dominant class membership, argmax(u_,axis=0)
    u0_ = initial cluster memberships
    d_ = distance matrix?
    jm_ =
    n_iter_ = number of iterations
    fpc_ = fuzzy partion coefficient (score)

    Fitted parameters are named with a trailing underscore.
    """

    def __init__(
            self,
            n_clusters=5, m=2, tol=1e-10, maxiter=1000,
            random_state=None,
            distance_metric='euclidean',
            predict_method='default'
        ):
        super(CmeansModel, self).__init__()
        self.n_clusters = n_clusters
        self.m = m
        self.tol = tol
        self.maxiter = maxiter
        self.random_state = check_random_state(random_state)
        self.distance_metric = distance_metric
        self.predict_method = predict_method
        
    def get_params(self, deep=False):
        # required for scikit-learn interoperability
        return {
            'n_clusters':self.n_clusters,
            'm':self.m,
            'tol':self.tol,
            'maxiter':self.maxiter,
            'distance_metric':self.distance_metric,
            'predict_method':self.predict_method
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_covariance(self, x):
        # compute covariance of hardened clusters
        return np.stack(
                [np.cov(x[self.hard_labels_==i].T) for i in range(self.n_clusters)]
            )
    
    def get_weighted_covariance(self, x):
        # compute covaraince weighted by cluster membership
        # WARNING @anla : this causes cumulative membership 
        # to bloat when the number of clusters is large
        return np.stack(
                [np.cov(x.T, aweights=x) for x in self.u_]
            )


    def fit(self, x, y=None):
        ''''\n    Fuzzy c-means clustering algorithm [1].\n\n    Parameters\n    ----------\n    data : 2d array, size (N, S)\n        Data to be clustered.  N is the number of data sets; S is the number\n        of features within each sample vector.\n    c : int\n        Desired number of clusters or classes.\n    m : float\n        Array exponentiation applied to the membership function u_old at each\n        iteration, where U_new = u_old ** m.\n    error : float\n        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.\n    maxiter : int\n        Maximum number of iterations allowed.\n    metric: string\n        By default is set to euclidean. Passes any option accepted by\n        ``scipy.spatial.distance.cdist``.\n    init : 2d array, size (S, N)\n        Initial fuzzy c-partitioned matrix. If none provided, algorithm is\n        randomly initialized.\n    seed : int\n        If provided, sets random seed of init. No effect if init is\n        provided. Mainly for debug/testing purposes.\n\n    Returns\n    -------\n    cntr : 2d array, size (S, c)\n        Cluster centers.  Data for each center along each feature provided\n        for every cluster (of the `c` requested clusters).\n    u : 2d array, (S, N)\n        Final fuzzy c-partitioned matrix.\n    u0 : 2d array, (S, N)\n        Initial guess at fuzzy c-partitioned matrix (either provided init or\n        random guess used if init was not provided).\n    d : 2d array, (S, N)\n        Final Euclidian distance matrix.\n    jm : 1d array, length P\n        Objective function history.\n    p : int\n        Number of iterations run.\n    fpc : float\n        Final fuzzy partition coefficient.\n\n\n    Notes\n    -----\n    The algorithm implemented is from Ross et al. [1]_.\n\n    Fuzzy C-Means has a known problem with high dimensionality datasets, where\n    the majority of cluster centers are pulled into the overall center of\n    gravity. If you are clustering data with very high dimensionality and\n    encounter this issue, another clustering method may be required. For more\n    information and the theory behind this, see Winkler et al. [2]_.\n\n    References\n    ----------\n    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.\n           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.\n\n    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high\n           dimensional spaces. 2012. Contemporary Theory and Pragmatic\n           Approaches in Fuzzy Computing Utilization, 1.\n    '''

        # check input
        x = check_array(x)

        # give model number of input features
        # as attribute for sklearn compatibility
        self.set_params(n_features_in_=x.shape[1])

        # try:
        cntr, u, u0, d, jm, p, fpc = cmeans(
            x.T, 
            self.n_clusters,
            self.m,
            error=self.tol,
            maxiter=self.maxiter,
            metric=self.distance_metric,
            seed=self.random_state.randint(2**32)
        )

        # for consistency
        # order clusters by 1st feature
        # cntr size (S,C)
        order = np.argsort(cntr[:, 0].ravel())

        self.cluster_centers_ = cntr[order, :]
        # self.u_ = u[order, :]
        self.labels_ = u[order, :]
        self.hard_labels_ = np.argmax(self.labels_, axis=0)
        self.u0_ = u0[order, :]
        self.d_ = d[order, :]
        self.jm_ = jm
        self.n_iter_ = p
        self.fpc_ = fpc

        # calculate covariance from training data
        # required for the chi2 method of predict
        self.cov_ = self.get_covariance(x)

        # except TypeError:
        #     raise TypeError("argument must be a string.* number")

        return self

    def predict(self, x, y=None, method='default', chi2_metric='euclidean', **kwargs):
        '''Prediction of new data in given a trained fuzzy c-means framework [1].
        Parameters

        ----------
        test_data : 2d array, size (S, N)
        New, independent data set to be predicted based on trained c-means
        from ``cmeans``. N is the number of features; S is the number of\n        features within each sample vector.\n    cluster_centers_trained : 2d array, size (S, c)\n        Location of trained centers from prior training c-means.\n    m : float\n        Array exponentiation applied to the membership function u_old at each\n        iteration, where U_new = u_old ** m.\n    error : float\n        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.\n    maxiter : int\n        Maximum number of iterations allowed.\n    metric: string\n        By default is set to euclidean. Passes any option accepted by\n        ``scipy.spatial.distance.cdist``.\n    init : 2d array, size (S, N)\n        Initial fuzzy c-partitioned matrix. If none provided, algorithm is\n        randomly initialized.\n    seed : int\n        If provided, sets random seed of init. No effect if init is\n        provided. Mainly for debug/testing purposes.\n\n    Returns\n    -------\n    u : 2d array, (S, N)\n        Final fuzzy c-partitioned matrix.\n    u0 : 2d array, (S, N)\n        Initial guess at fuzzy c-partitioned matrix (either provided init or\n        random guess used if init was not provided).\n    d : 2d array, (S, N)\n        Final Euclidian distance matrix.\n    jm : 1d array, length P\n        Objective function history.\n    p : int\n        Number of iterations run.\n    fpc : float\n        Final fuzzy partition coefficient.\n\n    Notes\n    -----\n    Ross et al. [1]_ did not include a predict algorithm to go along with\n    fuzzy c-means. This predict algorithm works by repeating the clustering\n    with fixed centers, then efficiently finds the fuzzy membership at all\n    points.\n\n    References\n    ----------\n    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.\n           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.'''

        # check if is check_is_fitted
        check_is_fitted(self, "cluster_centers_")

        # check input is OK
        x = check_array(x)
    
        if (method == 'default') & (self.predict_method=='default'):
            return cmeans_predict(
                x.T, 
                cntr_trained=self.cluster_centers_,
                m=self.m,
                error=self.tol,
                maxiter=self.maxiter,
                metric=self.distance_metric,
                seed=self.random_state.randint(2**32)
            )[0]

        elif (method == 'chi2') | (self.predict_method == 'chi2'):
            # compute the inverse covariance matrix
            if self.n_features_in_ == 1:
                vi = self.cov_
            else:
                vi = np.linalg.inv(self.cov_)

            return _chi2_predict(x, self.cluster_centers_, vi,
                metric=self.distance_metric, degrees_freedom='auto')

    def fit_predict(self, x, y=None, **kwargs):
        return self.fit(x).predict(x, **kwargs)

    def score(self, x=None, y=None):
        return self.fpc_
