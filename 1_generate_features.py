import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


logFormatter = logging.Formatter("%(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

from PASDAC import SETTINGS
from PASDAC.Preprocessing import smooth
from PASDAC.Segmentation import segment
from PASDAC.Features import get_features
from PASDAC.Tools import interval_intersect_interval, create_folder
from PASDAC.Tools.util import timeToIndex
from PASDAC.Tools import save_data
from PASDAC.Tools.asserting import remove_duplicate_segmentation


create_folder(SETTINGS['PATH_OUTPUT'])

fileHandler = logging.FileHandler(os.path.join(SETTINGS['PATH_OUTPUT'], "features.log"))
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

dataAll = {}


for subj in SETTINGS['SUBJECT_LIST']:
    logger.info("=========================================")
    logger.info("Processing data of Subject %s", subj)

    subjData = {}

    # read data
    dataFileName = "{}/subject{}_{}_data.csv".format(
        SETTINGS['PATH_DATA'], subj, SETTINGS['DATASET'])
    labelsFileName = "{}/subject{}_{}_label.csv".format(
        SETTINGS['PATH_DATA'], subj, SETTINGS['DATASET'])

    logger.info(dataFileName)
    logger.info(labelsFileName)

    # import data file
    if os.path.isfile(dataFileName):
        data = pd.read_csv(dataFileName)
    else:
        logger.exception('fileDoesNotExist, ' +
                         dataFileName + ' does not exist in the file system.')

    # import label file
    if os.path.isfile(labelsFileName):
        label = pd.read_csv(labelsFileName)
    else:
        logger.exception('fileDoesNotExist, ' +
                         labelsFileName + ' does not exist in the file system.')

    subDataThreshForPlot = 1000
    
    plt.title('Before smoothing')
    if (data.shape[0] > subDataThreshForPlot):
        plt.plot(data[SETTINGS['SENSORS_AVAILABLE']].iloc[0:subDataThreshForPlot])
    else:
        plt.plot(data[SETTINGS['SENSORS_AVAILABLE']])
    plt.show()
    

    # ============== smoothing data =================
    #smoothedData = smooth(data)
    smoothedData = data
    plt.title('After smoothing')
    if (data.shape[0] > subDataThreshForPlot):
        plt.plot(smoothedData[SETTINGS['SENSORS_AVAILABLE']].iloc[0:subDataThreshForPlot])
    else:
        plt.plot(smoothedData[SETTINGS['SENSORS_AVAILABLE']])
    plt.show()


    # ============= segmentation algorithm ===========
    segmentation_in_time = segment(smoothedData)
    segmentation_in_index = pd.DataFrame({'Start': timeToIndex(segmentation_in_time['Start'], data['Timestamp (ms)'].as_matrix()),
                                          'End':   timeToIndex(segmentation_in_time['End'], data['Timestamp (ms)'].as_matrix())},
                                         columns=['Start', 'End'])

    # removing the segments that cross over discontinuous data
    index_distinct = remove_duplicate_segmentation(segmentation_in_index)
    segmentation_in_index = segmentation_in_index.iloc[index_distinct, :].reset_index(drop=True)
    segmentation_in_time = segmentation_in_time.iloc[index_distinct, :].reset_index(drop=True)

    
    label_segmentation, index_covered = interval_intersect_interval(
        segmentation=segmentation_in_time, groundtruth=label)
  #  label_segmentation = np.array(label_segmentation)

    # slicing the segments that have overlapping label
    segmentation_in_time = segmentation_in_time.iloc[index_covered,:].reset_index(drop=True)
    segmentation_in_index = segmentation_in_index.iloc[index_covered,:].reset_index(drop=True)

    # ============= feature extraction ======================
    feature, feature_type, feature_description = get_features(
        smoothedData, segmentation_in_index)
    # drop NaN
   # label_segmentation = pd.Series(label_segmentation, index = ['label'])
    label_segmentation = pd.DataFrame(label_segmentation)
    feature_ = pd.concat([feature,label_segmentation,segmentation_in_time], axis=1)
  #  label_segmentation = np.array(label_segmentation)
    feature_ = feature_.dropna(axis=0, how='any')

    # ============= put into dataAll ========================
    subjData['data'] = smoothedData
    subjData['feature'] = feature_.iloc[:,0:13]
    subjData['groundtruth'] = label
#    subjData['segmentation_in_time'] = segmentation_in_time
    subjData['segmentation_in_time'] = feature_.iloc[:,14:16]
    subjData['segmentation_in_index'] = segmentation_in_index
 #   subjData['label_segmentation'] = label_segmentation
    subjData['label_segmentation'] = feature_.iloc[:,13:14]
    dataAll[subj] = subjData


## HRV
#    subjData['data'] = smoothedData
#    subjData['feature'] = feature_.iloc[:,8:13]
#    subjData['groundtruth'] = label
##    subjData['segmentation_in_time'] = segmentation_in_time
#    subjData['segmentation_in_time'] = feature_.iloc[:,14:16]
#    subjData['segmentation_in_index'] = segmentation_in_index
# #   subjData['label_segmentation'] = label_segmentation
#    subjData['label_segmentation'] = feature_.iloc[:,13:14]
#    dataAll[subj] = subjData


save_data(dataAll)
