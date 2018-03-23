"""Feature extraction algorithms

"""
import logging
import pandas as pd
from PASDAC.settings import SETTINGS
from .calculateFeaturesVerySimple import calculateFeaturesVerySimple
from .calculateECGFeatures import calculateECGFeatures

logger = logging.getLogger(__name__)


def get_features(data, segmentation_in_index):
    """Master feature extraction

    Parameters
    ----------
        data:                   dataFrame
        segmentation_in_index:  dataFrame representing segmentation

    Return
    ------
        featuresDf:     dataFrames
        fType:          list
                        eg: ['Mean_acc_1_x', 'Variance_acc_1_x', 'Mean_acc_1_y', 'Variance_acc_1_y',
                            ....
                           'Mean_gyr_3_y', 'Variance_gyr_3_y']
        fDescr:         list
                        eg: ['Mean', 'Variance']
    """

    if SETTINGS['FEATURE_TYPE'] == 'VerySimple':

        logger.info("Feature type: %s", SETTINGS['FEATURE_TYPE'])

        fDescr = ['Mean', 'Variance']
        sensorList = SETTINGS['SENSORS_AVAILABLE']

        data = data[sensorList]

        fType = []

        for s in sensorList:
            for f in fDescr:
                fType.extend([f + "_" + s])

        allfeats = []

        N = segmentation_in_index.shape[0]

        logger.info("Number of segments %d", N)

        for i in range(N):

            if i % 1000 == 0:
                logger.info("Processing %d%% of segments", 100 * i / N)

            chunk = data.iloc[segmentation_in_index.Start.iloc[i]:segmentation_in_index.End.iloc[i]]
            chunk = chunk.as_matrix()

            features = []

            for s in sensorList:
                f = calculateFeaturesVerySimple(chunk)
                features.extend(f)

            allfeats.append(features)

        featuresDf = pd.DataFrame(data=allfeats, columns=fType)

        
    elif SETTINGS['FEATURE_TYPE'] == 'ecgFeatures':
        

        logger.info("Feature type: %s", SETTINGS['FEATURE_TYPE'])

        fDescr = [ 's_mean', 's_med', 's_max', 's_var', 's_std_dev', 's_abs_dev', 's_kurtois', 's_skew',
#         'power_ulf',
        'power_vlf','power_lf','power_hf','lf_hf_ratio'
         ,'rr_interval'
         ]
        sensorList = SETTINGS['SENSORS_AVAILABLE']

        time = data['Timestamp (ms)']
        data = data['Sample (V)']
        fType = []

        for s in sensorList:
            for f in fDescr:
                fType.extend([f + "_" + s])

        allfeats = []

        N = segmentation_in_index.shape[0]

        logger.info("Number of segments %d", N)

        for i in range(N):

            if i % 1000 == 0:
                logger.info("Processing %d%% of segments", 100 * i / N)

            chunk = data.iloc[segmentation_in_index.Start.iloc[i]:segmentation_in_index.End.iloc[i]]
            chunk_time = time.iloc[segmentation_in_index.Start.iloc[i]:segmentation_in_index.End.iloc[i]]

            features = []
            chunk_df = pd.DataFrame({'Sample (V)':chunk,'Timestamp (ms)':chunk_time})
            for s in sensorList:

                f = calculateECGFeatures(chunk_df)
                features.extend(f)

            allfeats.append(features)

        featuresDf = pd.DataFrame(data=allfeats, columns=fType)

        
        
    
    else:

        raise ValueError("Not implemented")

    return featuresDf, fType, fDescr
