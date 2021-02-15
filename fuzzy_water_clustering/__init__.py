""""
fuzzy_water_clustering, a scikit-learn compatible fuzzy cmeans (FCM) toolbox.

This package provides cmeans estimator objects for unsupervised clustering of data, along with scoring metrics that can be used with scikit-learn. Examples and guides are included in the ./doc/guides folder.

Recommended Use
---------------
>>> import skfuzzy as fuzz

"""

__all__ = []

######################
# Subpackage imports #
######################

# Core clustering estimator objects
import fuzzy_water_clustering.utils as _utils  # noqa: E402
from fuzzy_water_clustering.utils import *  # noqa: E402,F403
__all__.extend(_utils.__all__)

# Fuzzy membership function subpackage
import fuzzy_water_clustering.estimators as _estimators  # noqa: E402
from fuzzy_water_clustering.estimators import *  # noqa: E402,F403
__all__.extend(_estimators.__all__)
