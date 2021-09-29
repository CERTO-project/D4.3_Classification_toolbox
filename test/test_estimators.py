'''
pytest script for scikit-learn estimators

AUTHOR:
    Angus Laurenson, Plymouth Marine Laboratory
    anla@pml.ac.uk

DESCRIPTION:
    This script tests estimators against scikit-learn
    standards using sklearn.utils.estimator_checks
    which adds tests depending on what sklearn Base
    or MixIn classes the custom estimators inherit.
    
    It is designed for use with pytest and will be
    automatically picked up by pytest command.

USAGE:
    # Call all test scrips using pytest, including
    # this one:
    >> pytest

    # Specifcally call this script with pytest:
    >> pytest test_estimator.py

'''

import pytest

# setup logger
import logging
logger = logging.getLogger(name='CmeansModel_tester')

# import the package locally to test
import sys; sys.path.append("..")
import fuzzy_water_clustering as fwc

# import an extensive suite of checks
# from scikit-learn
from sklearn.utils import estimator_checks

# list of estimators to check
ESTIMATORS = [fwc.CmeansModel()]

@estimator_checks.parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)

# FIXME @anla : no test for CmeansModel.predict(method='chi2')
