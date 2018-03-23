import os
import logging
import sys
import numpy as np
import pandas as pd
from PASDAC.settings import SETTINGS
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from PASDAC.Tools import interval_intersect_interval
from PASDAC.Classification import classification
from PASDAC.Tools.util import segmentsToTimeSeries, timeToIndex
from PASDAC.Plot import plotROC, plotConfusionMatrix
from PASDAC.Preprocessing.standardizing import standardizeData
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def run_validation(dataAll):
    """ Split data depending on SETTINGS and run evaluation
    Parameters
    ----------
        dataAll:                dict
    """
    partition_setting = SETTINGS['PARTITION']
    logger.info(partition_setting)
    returnvals = {}

    if partition_setting['SUBJECT_DEPENDENCY'] == 'personalized':
        """
        Personalized Evaluation
        """

        logger.info("Personalized evaluation")

        for subject in dataAll:

            logger.info("Personalized evaluation of subject {} (equally splitting)".format(subject))

            feature = dataAll[subject]['feature'].as_matrix()
            groundtruth = dataAll[subject]['groundtruth']
            segmentation_in_index = dataAll[subject]['segmentation_in_index']
            segmentation_in_time = dataAll[subject]['segmentation_in_time']
            label_segmentation = dataAll[subject]['label_segmentation']

            # Equally splitting:
            # Segmentations that belong in the same chunk of activites,
            # are not splitted into train and test set, since they represent
            # different part of an activity

            chunk_groundtruth = groundtruth.copy()
            chunk_groundtruth['Label'] = np.arange(chunk_groundtruth.shape[0])
            chunk, index_covered = interval_intersect_interval(segmentation=segmentation_in_time, 
                                                groundtruth=chunk_groundtruth)
                
            mapping_chunk_to_label = {}
            for i in range(len(chunk)):
                val = chunk[i]
                if val in mapping_chunk_to_label:
                    continue
                else:
                    mapping_chunk_to_label.update({val: label_segmentation[i]})

            unique_chunks = list(mapping_chunk_to_label.keys())
            unique_labels = list(mapping_chunk_to_label.values())

            trainchunks, testchunks, y_train, y_test = train_test_split(
                unique_chunks, unique_labels, test_size=0.5, stratify=unique_labels)

            trainData = []
            testData = []
            trainLabel = []
            testLabel = []
            test_index = []
            for val in range(len(chunk)):
                if chunk[val] in trainchunks:
                    trainData.append(feature[val, :])
                    trainLabel.append(label_segmentation[val])

                if chunk[val] in testchunks:
                    testData.append(feature[val, :])
                    testLabel.append(label_segmentation[val])
                    test_index.append(val)

            trainData = np.vstack(trainData)
            trainLabel = np.array(trainLabel)
            testData = np.vstack(testData)
            testLabel = np.array(testLabel)
            
            trainData, testData = standardizeData(trainData, testData)
            prediction, probability = classification(trainData, trainLabel, testData)
            
            train=pd.DataFrame(trainData)
            train.to_csv('file_name.csv', sep='\t')

            logger.info("Plotting ROC")
            saving_path = os.path.join(SETTINGS['PATH_OUTPUT'], '{}_personalized.png'.format(subject))
            plotROC(testLabel, probability.as_matrix(), len(SETTINGS['CLASS_LABELS']), saving_path)
            logger.info("Confusion matrix for segments:")
            confusion = confusion_matrix(testLabel, prediction)
            logger.info(confusion)
            f1 = f1_score(testLabel, prediction, average='weighted')  
            logger.info(f1)
            auc = roc_auc_score(testLabel, prediction, average='weighted')
            logger.info(auc)
            w = np.ones(testLabel.shape[0])
            logger.info(accuracy_score(testLabel, prediction, normalize=True,sample_weight = w))
            
    elif partition_setting['SUBJECT_DEPENDENCY'] == 'generalized':
        """ Leave One Subject Out Evaluation
        """

        if len(dataAll) == 1:
            raise ValueError("Cannot perform LOSO when number of subjects is 1")

        logger.info("Generalized evaluation")

        count = 0

        for test_subj in dataAll:
            if(count == 1) :
                break
            logger.info("Leave subject {} out".format(test_subj))

            data = dataAll[test_subj]['data']
            groundtruth = dataAll[test_subj]['groundtruth']
            segmentation_in_time = dataAll[test_subj]['segmentation_in_time'].astype(int)
            segmentation_in_index = dataAll[test_subj]['segmentation_in_index']
            feature = dataAll[test_subj]['feature'].as_matrix()
            label_segmentation = dataAll[test_subj]['label_segmentation']

            trainData = []
            trainLabel = []
            for train_subj in SETTINGS['SUBJECT_LIST']:
                if train_subj == test_subj:
                    continue
                trainData.append(dataAll[train_subj]['feature'])
                trainLabel.append(dataAll[train_subj]['label_segmentation'])

            trainData = np.concatenate(trainData)
            trainLabel = np.concatenate(trainLabel)

            testData = feature
            testLabel = label_segmentation

            trainData, testData = standardizeData(trainData, testData)
            prediction, probability = classification(trainData, trainLabel, testData)

            logger.info("Plotting ROC")
            saving_path = os.path.join(SETTINGS['PATH_OUTPUT'], '{}_generalized.png'.format(test_subj))
            plotROC(testLabel, probability.as_matrix(), len(SETTINGS['CLASS_LABELS']), saving_path)
            logger.info("Confusion matrix for segments:")
            confusion = confusion_matrix(testLabel, prediction)
            logger.info(confusion)
            f1 = f1_score(testLabel, prediction, average='weighted')  
            logger.info(f1)
#            auc = roc_auc_score(testLabel, prediction, average='weighted')
      #      logger.info(auc)
            w = np.ones(testLabel.shape[0])
            logger.info(accuracy_score(testLabel, prediction, normalize=True,sample_weight = w))
            count = count + 1