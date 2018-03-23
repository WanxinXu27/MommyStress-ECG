"""Standardizing algorithms

This is used for both raw data and features
"""

import logging
from scipy.stats import zscore
import pandas as pd
import numpy as np
from PASDAC.settings import SETTINGS

logger = logging.getLogger(__name__)


def standardizeData(trainingData, testData):
    """training set and test set z-normalization without taking use of scipy library

    Parameters
    ----------
        trainingData:           dataFrame
        testData:               dataFrame

    Return
    ------
        trainingZsc:            dataFrame
        testZsc:                dataFrame

    """

    # TODO: Comment in the following 2 lines and implement training set and test set z-normalization 
    #       according to requirement
    
    #  training set z-normalization

    # stats.zscore can take dataframe as input
    trainingZsc =  zscore(trainingData)

    trainingMean = np.mean(trainingData, axis=0)
    trainingStd = np.std(trainingData, axis=0)
    testZsc = testData.copy()

    for i in range(trainingData.shape[1]):
        testZsc[:,i] = testZsc[:,i] - trainingMean[i]
        testZsc[:,i] = testZsc[:,i]/trainingStd[i]

    # # HACK: to avoid null-division: set stdvar==0 to stdvar=mean
    # if 0 in trainingStd:
    #     ind = np.where((trainingStd==0))[0]
    #     for i in ind:
    #         trainingZsc.ix[:,i] = trainingData.ix[:,i] - trainingMean[i]
    #         testZsc.ix[:,i] = testData.ix[:,i] - trainingMean[i]



    return trainingZsc, testZsc