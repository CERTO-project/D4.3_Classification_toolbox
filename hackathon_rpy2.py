#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:30:19 2020

@author: liz

Use R cmeans function from python, roughly following:
    https://community.alteryx.com/t5/Data-Science/RPy2-Combining-the-Power-of-R-Python-for-Data-Science/ba-p/138432
    https://rpy2.github.io/doc/v3.0.x/html/robjects_functions.html
"""

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

rnorm=robjects.r('rnorm')
rnorm(100)

e1071 = importr('e1071')
e1071 = importr('~/anaconda3/envs/cran/lib/R/library/e1071/R/e1071.rdx')

stats = importr('stats')



plot = graphics.plot
rnorm = stats.rnorm
plot(rnorm(100), ylab="random") 

