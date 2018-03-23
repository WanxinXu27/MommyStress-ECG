from __future__ import division
import logging
import numpy as np
from intervaltree import Interval, IntervalTree


logger = logging.getLogger(__name__)


def _get_overlap(a, b):
    """Find intersetion between two pairs a and b

    Parameters
    ----------
    a:          pair (support a[0] and a[1])
    b:          pair (support b[0] and b[1])

    Return:
    -------
    overlap:    overlap between two intervals
    """

    overlap = min(a[1], b[1]) - max(a[0], b[0])
    return max(0, overlap)


def interval_intersect_interval(**kwargs):
    """Determine label of each segmentation based on
    intersection with labels
    
    Parameters
    ----------
    groundtruth:      dataframe containing columns 'Start', 'End', 'Label'
    segmentation:     dataframe containing columns 'Start', 'End'

    Return:
    -------
    label_segmentation:   list of labels, same number of rows as segmentation
        represent label of each segment

    """

    label_segmentation = []

    gt = kwargs['groundtruth']
    segmentation = kwargs['segmentation']

    tree = IntervalTree()
    for i in range(gt.shape[0]):
        tree.add(Interval(gt['Start'][i],gt['End'][i], i))


    index_covered = []

    for i in range(segmentation.shape[0]):
        interval = (segmentation.Start.iloc[i], segmentation.End.iloc[i])

        overlapping_labels = sorted(tree.search(interval[0], interval[1]))

        if len(overlapping_labels) == 0:
            logger.debug("A segment does not have overlapping groundtruth")

        elif len(overlapping_labels) == 1:
            # only one overlapping label
            label_segmentation.append(gt['Label'].iloc[overlapping_labels[0].data])
            index_covered.append(i)

        else:
            # majority voting:
            # if there are multiple overlapping labels,
            # select the one with largest overlap

            overlap_time = []
            for label in overlapping_labels:
                overlap_time.append(_get_overlap(label, interval))
        
            index_max = np.argmax(np.array(overlap_time))

            label_segmentation.append(gt['Label'].iloc[overlapping_labels[index_max].data])
            index_covered.append(i)
#    logger.info(label_segmentation)
    logger.info("Percentage of segments that do not have label: {}%".format(\
        100*(1 -len(index_covered)/len(label_segmentation))))

    return label_segmentation, index_covered
