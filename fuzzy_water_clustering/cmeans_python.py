'''
# CmeansModel
A scikit-learn compatible cmeans clustering estimator object.
Built on skfuzzy.cluster.cmeans and cmeans_predict functions.

`from cmeans_python import CmeansModel`

See class docstring for more info.

Author : Angus Laurenson
Email  : anla@pml.ac.uk
'''

# cmeans algorithms
from skfuzzy.cluster import cmeans, cmeans_predict

# for data handling
import numpy as np

# scikit-learn checks
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

# scikit-learn base classes
from sklearn.base import BaseEstimator, ClusterMixin

# distances
from scipy.spatial.distance import cdist
from scipy.stats import chi2

class CmeansModel(BaseEstimator, ClusterMixin):
    """A c-means clustering estimator that is compatible with scikit-learn.

    Built on skfuzzy.cluster.cmeans and cmeans_predict functions.
    fit() and predict() methods have original docstrings

    # USAGE #

    cmeans = CmeansModel(
        c=5,
        m=2,
        err=0.005,
        maxiter=1000,
        random_state=None,
        random_starts=1,
        scoring_metric='fpc'
        distance_metric='euclidean'
    )

    # determine clusters
    cmeans.fit(X)

    # estimate memberships
    lables = cmeans.predict(X)

    # score clustering
    score = cmeans.score()

    # INIT ARGUMENTS #

    c is int number of clusters
    m is fuzziness factor ~ 1.5 < M < 2.5
    err is error threshold
    maxiter is maximum number of iteration
    random_state is for scikit-learn learn not sure
    scoring_metric is the name of method to use when scoring, used for finding optimum parameters

    # METHODS #

    fit(X)
    predict(X)
    fit_predict(X)
    score()

    where X is an array of size (N,M),
    N is the number of observations,
    M is the number of of features.

    # ATTRIBUTES #

    cntr_ = cluster centre coordinates
    u_ = array of cluster memberships
    labels_ = dominant class membership, argmax(u_,axis=0)
    u0_ = initial cluster memberships
    d_ =
    jm_ =
    p_ =
    fpc_ = fuzzy partion coefficient (score)

    """

    def __init__(
            self,
            c=5, m=2, err=0.005, maxiter=1000,
            random_state=None,
            scoring_metric='fpc',
            distance_metric='euclidean'
        ):
        super(CmeansModel, self).__init__()
        self.c = c
        self.m = m
        self.err = err
        self.maxiter = maxiter
        self.random_state = check_random_state(random_state)
        self.scoring_metric = scoring_metric
        self.distance_metric = distance_metric

    def get_params(self, deep=False):
        # required for scikit-learn interoperability
        return {
            'c':self.c,
            'm':self.m,
            'err':self.err,
            'maxiter':self.maxiter,
            'distance_metric':self.distance_metric
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y=None):
        ''''\n    Fuzzy c-means clustering algorithm [1].\n\n    Parameters\n    ----------\n    data : 2d array, size (N, S)\n        Data to be clustered.  N is the number of data sets; S is the number\n        of features within each sample vector.\n    c : int\n        Desired number of clusters or classes.\n    m : float\n        Array exponentiation applied to the membership function u_old at each\n        iteration, where U_new = u_old ** m.\n    error : float\n        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.\n    maxiter : int\n        Maximum number of iterations allowed.\n    metric: string\n        By default is set to euclidean. Passes any option accepted by\n        ``scipy.spatial.distance.cdist``.\n    init : 2d array, size (S, N)\n        Initial fuzzy c-partitioned matrix. If none provided, algorithm is\n        randomly initialized.\n    seed : int\n        If provided, sets random seed of init. No effect if init is\n        provided. Mainly for debug/testing purposes.\n\n    Returns\n    -------\n    cntr : 2d array, size (S, c)\n        Cluster centers.  Data for each center along each feature provided\n        for every cluster (of the `c` requested clusters).\n    u : 2d array, (S, N)\n        Final fuzzy c-partitioned matrix.\n    u0 : 2d array, (S, N)\n        Initial guess at fuzzy c-partitioned matrix (either provided init or\n        random guess used if init was not provided).\n    d : 2d array, (S, N)\n        Final Euclidian distance matrix.\n    jm : 1d array, length P\n        Objective function history.\n    p : int\n        Number of iterations run.\n    fpc : float\n        Final fuzzy partition coefficient.\n\n\n    Notes\n    -----\n    The algorithm implemented is from Ross et al. [1]_.\n\n    Fuzzy C-Means has a known problem with high dimensionality datasets, where\n    the majority of cluster centers are pulled into the overall center of\n    gravity. If you are clustering data with very high dimensionality and\n    encounter this issue, another clustering method may be required. For more\n    information and the theory behind this, see Winkler et al. [2]_.\n\n    References\n    ----------\n    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.\n           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.\n\n    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high\n           dimensional spaces. 2012. Contemporary Theory and Pragmatic\n           Approaches in Fuzzy Computing Utilization, 1.\n    '''

        # check input
        X = check_array(X)

        try:
            cntr, u, u0, d, jm, p, fpc = cmeans(
                X.T, self.c, self.m, error=self.err, maxiter=self.maxiter,
                metric=self.distance_metric, seed=self.random_state.randint(1000)
            )

            # for consistency
            # order clusters by 1st feature
            # cntr size (S,C)
            order = np.argsort(cntr[:, 0].ravel())

            self.cntr_ = cntr[order, :]
            self.u_ = u[order, :]
            self.labels_ = np.argmax(self.u_, axis=0)
            self.u0_ = u0[order, :]
            self.d_ = d[order, :]
            self.jm_ = jm
            self.p_ = p
            self.fpc_ = fpc

            # calculate covariance from training data
            self.cov_ = np.stack(
                [np.cov(X.T, aweights=x) for x in self.u_]
            )

        except TypeError:
            raise TypeError("argument must be a string.* number")

        # sort the results in some way to have consistency....

        return self

    def predict(self, X, y=None, method='default'):
        ''''\n    Prediction of new data in given a trained fuzzy c-means framework [1].\n\n    Parameters\n    ----------\n    test_data : 2d array, size (S, N)\n        New, independent data set to be predicted based on trained c-means\n        from ``cmeans``. N is the number of data sets; S is the number of\n        features within each sample vector.\n    cntr_trained : 2d array, size (S, c)\n        Location of trained centers from prior training c-means.\n    m : float\n        Array exponentiation applied to the membership function u_old at each\n        iteration, where U_new = u_old ** m.\n    error : float\n        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.\n    maxiter : int\n        Maximum number of iterations allowed.\n    metric: string\n        By default is set to euclidean. Passes any option accepted by\n        ``scipy.spatial.distance.cdist``.\n    init : 2d array, size (S, N)\n        Initial fuzzy c-partitioned matrix. If none provided, algorithm is\n        randomly initialized.\n    seed : int\n        If provided, sets random seed of init. No effect if init is\n        provided. Mainly for debug/testing purposes.\n\n    Returns\n    -------\n    u : 2d array, (S, N)\n        Final fuzzy c-partitioned matrix.\n    u0 : 2d array, (S, N)\n        Initial guess at fuzzy c-partitioned matrix (either provided init or\n        random guess used if init was not provided).\n    d : 2d array, (S, N)\n        Final Euclidian distance matrix.\n    jm : 1d array, length P\n        Objective function history.\n    p : int\n        Number of iterations run.\n    fpc : float\n        Final fuzzy partition coefficient.\n\n    Notes\n    -----\n    Ross et al. [1]_ did not include a prediction algorithm to go along with\n    fuzzy c-means. This prediction algorithm works by repeating the clustering\n    with fixed centers, then efficiently finds the fuzzy membership at all\n    points.\n\n    References\n    ----------\n    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.\n           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.\n    '''

        # check if is check_is_fitted
        check_is_fitted(self, "cntr_")

        # check input is OK
        X = check_array(X)

        if method == 'default':

            return cmeans_predict(
                X.T, cntr_trained=self.cntr_,
                m=self.m,
                error=self.err,
                maxiter=self.maxiter,
                metric=self.distance_metric
            )[0]

        if method == 'chi2':
            """predict membership from covariance and mean of class"""

            n_features = self.cntr_.shape[1]

            dist = cdist(
                X,
                np.atleast_2d(self.cntr_),
                metric=self.distance_metric,
                VI=np.linalg.inv(self.cov_)
            )

            memberships = 1 - chi2.cdf(dist**2, n_features)

            return memberships.T

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def score(self, X=None, y=None):
        return self.fpc_
