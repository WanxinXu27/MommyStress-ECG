import numpy as np
import pandas as pd
from PASDAC.settings import SETTINGS
import math


def segmentsToTimeSeries(segments_index, prediction, label):
    """ convert prediction of segments into time series
    last come will overwrite the label
    will implement majority voting later

    """
    max_index = segments_index.as_matrix().max()

    sensor_label = np.zeros((max_index + 1, ))
    sensor_prediction = np.zeros((max_index + 1, ))

    sensor_label.fill(np.nan)
    sensor_prediction.fill(np.nan)

    for i in range(segments_index.shape[0]):
        sensor_label[segments_index['Start'].iloc[i]:
                     segments_index['End'].iloc[i] + 1] = label[i]

        sensor_prediction[segments_index['Start'].iloc[i]:
                          segments_index['End'].iloc[i] + 1] = prediction[i]

    sensor_label_removegap = [i for i in sensor_label if not math.isnan(i)]
    sensor_prediction_removegap = [
        i for i in sensor_prediction if not math.isnan(i)]

    return np.array(sensor_pxwrediction_removegap), np.array(sensor_label_removegap)


def timeToIndex(unixtimes, sensortime):
    """Given array of Unix time, 
    and sorted ['Time'] array from sensor data
    find corresponding index in sensor data
    """
    index = []
    for unixtime in unixtimes:
        index.append(np.searchsorted(sensortime, unixtime))

    return np.array(index)
