import logging
from numpy.testing import assert_array_equal
from PASDAC.settings import SETTINGS


logger = logging.getLogger()


def assert_label(label):
	"""Check the convention that label is continuous and start from 0
	"""

	convention = np.arange(len(SETTINGS['CLASS_LABEL']))

	assert_array_equal(convention, np.unique(label))


def remove_duplicate_segmentation(segmentation_in_index):

	index_distinct = []

	for i in range(segmentation_in_index.shape[0]):
		if segmentation_in_index['Start'][i] == segmentation_in_index['End'][i]:
			logger.debug("Segmentation start and end are equal, data collection could be not continuous")
		else:
			index_distinct.append(i)


	return index_distinct