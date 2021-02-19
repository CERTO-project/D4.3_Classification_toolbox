'''
# CmeansModel from CRAN package e1071
A scikit-learn compatible cmeans clustering estimator object.
Reference: https://scikit-learn.org/stable/developers/develop.html
Built on e1071.cmeans function.

`from cmeans_cran import CmeansCRAN`

See class docstring for more info.

Author : Liz Atwood
Email  : liat@pml.ac.uk
'''

# CRAN package objects
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

e1071 = importr('e1071')
cmeans = e1071.cmeans

# scikit-learn checks REMOVE?
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

# scikit-learn scoring metric REMOVE?
from sklearn.metrics import silhouette_score

# scikit-learn base classes
from sklearn.base import BaseEstimator, ClusterMixin

# distances
from scipy.spatial.distance import cdist
from scipy.stats import chi2

class CmeansCRAN(BaseEstimator, ClusterMixin):
    """The CRAN c-means clustering estimator, compatible with scikit-learn.

    Built on e1071.cmeans function.
    fit() and predict() methods have original docstrings

    # USAGE #

    cmeans = CmeansCRAN(
        centers=5,
        m = 2,
        iter.max = 100,
        verbose = FALSE,
        dist = "euclidean",
        method = "cmeans",
        rate.par = NULL,
        weights = 1,
        control = list(),
        random_state=None
    )

    # determine clusters
    cmeans.fit(X)

    # estimate memberships
    lables = cmeans.predict(X)

    # score clustering
    score = cmeans.score()

    # INIT ARGUMENTS #

    centers is int number of clusters
    m is fuzziness factor ~ 1.5 < M < 2.5
    iter.max is maximum number of iterations
    random_state is for scikit-learn learn not sure
    no scoring_metric in call, only uses objective function

    # METHODS #

    fit(X)
    predict(X)
    fit_predict(X)
    score()

    where X is an array of size (N,M),
    N is the number of observations,
    M is the number of features.

    # ATTRIBUTES #

    centers_ = cluster centre coordinates
    size_ = number data points in each cluster of closest hard clustering
    cluster_ = int vector cluster indices where data points assigned to closest hard clustering (max membership)
    iter_ = number of iterations performed
    membership_ = matrix with membership values of data points to clusters
    withinerror_ = value of objective function
    call_ = call used to create the object

    """

    def __init__(
            self,
            centers=5, m=2, iter.max = 100,
            verbose = FALSE,
            dist = "euclidean",
            method = "cmeans",
            rate.par = NULL,
            weights = 1,
            control = list(),
            random_state=None
        ):
        super(CmeansModel, self).__init__()
        self.centers = centers
        self.m = m
        self.iter.max = iter.max
        self.verbose = verbose
        self.dist = dist
        self.method = method
        self.rate.par = rate.par
        self.weights = weights
        self.control = control
        self.random_state = check_random_state(random_state)

    def get_params(self, deep=False):
        # required for scikit-learn interoperability
        return {
            'centers':self.centers,
            'm':self.m,
            'iter.max':self.iter.max,
            'verbose':self.verbose,
            'dist':self.dist,
            'method':self.method,
            'rate.par':self.rate.par,
            'weights':self.weights,
            'control':self.control
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y=None):
        ''''https://rdrr.io/rforge/e1071/man/cmeans.html'''

        # check input
        X = check_array(X)

        try:
            centers, membership, iter = cmeans(
                X.T, self.centers, self.m, maxiter=self.iter.max,
                metric=self.dist
            )

            # for consistency
            # order clusters by 1st feature
            # cntr size (S,C)
            order = np.argsort(centers[:, 0].ravel())

            self.centers_ = centers[order, :]
            self.u_ = u[order, :]
            self.labels_ = np.argmax(self.u_, axis=0)
            self.u0_ = u0[order, :]
            self.d_ = d[order, :]
            self.jm_ = jm
            self.p_ = p
            self.fpc_ = fpc
            self.silouette_score_ = silhouette_score(X, self.labels_)

            # calculate covariance from training data
            self.cov_ = np.stack(
                [np.cov(X.T, aweights=x) for x in self.u_]
            )

        except TypeError:
            raise TypeError("argument must be a string.* number")

        # sort the results in some way to have consistency....

        return self
