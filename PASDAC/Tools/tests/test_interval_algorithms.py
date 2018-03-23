import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from PASDAC.Tools.interval_algorithms import _get_overlap, interval_intersect_interval


def test_get_overlap_1():
	a = [10, 20]
	b = [15, 30]

	assert _get_overlap(a, b) == 5


def test_get_overlap_2():
	a = [10, 20]
	b = [30, 40]

	assert _get_overlap(a, b) == 0


def test_interval_intersect_interval():

	gt = pd.DataFrame({'Start': [1,4,6,8], 'End':[4,6,8,10], 'Label': [10,11,12,13]})

	pred_start = np.arange(1,10, 0.5)
	pred_end = pred_start + 1

	pred = pd.DataFrame({'Start':pred_start, 'End':pred_end})

	label = interval_intersect_interval(groundtruth=gt, segmentation=pred)

	assert_array_equal(label, np.array([10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]))


def test_interval_intersect_interval_2():

	gt = pd.DataFrame({'Start': [1,4,6,8], 'End':[4,6,8,10], 'Label': [10,11,12,13]})

	pred_start = np.arange(1,10, 0.5)
	pred_end = pred_start + 1

	pred = pd.DataFrame({'Start':pred_start, 'End':pred_end})

	label = interval_intersect_interval(groundtruth=gt, segmentation=pred)

	assert_array_equal(label, np.array([10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]))



