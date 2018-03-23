from PASDAC.Tools.util import segmentsToTimeSeries, timeToIndex
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal


# def test_segmentToTimeSeries():
# 	segments = np.array([[0,4],[3,5],[4,8]])
# 	segments_index = pd.DataFrame(data=segments,columns=['Start','End'],dtype=int)

# 	label = np.array([10,11,12])

# 	timeseries = segmentsToTimeSeries(segments_index, label)

# 	assert_array_equal(timeseries.astype(int), np.array([10,10,10,11,12,12,12,12,12]))


def test_timeToIndex():

	sensorTime = np.arange(10, 1000, 20)

	unixTimes = [25,35]

	indices = timeToIndex(unixTimes, sensorTime)

	assert_array_equal(indices, np.array([1,2]))
	