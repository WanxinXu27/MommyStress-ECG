"""Utilities for file system handling

"""
import os
from os.path import join, exists
import logging
import pickle
import numpy as np
import pandas as pd
import shutil
from PASDAC.settings import SETTINGS

logger = logging.getLogger(__name__)


def create_folder(folderpath, delete_existing=False):
    '''
    Create the folder
    Parameters:
            f: folder path. Could be nested path (so nested folders will be created)
            deleteExising: if True then the existing folder will be deleted.
    '''
    if exists(folderpath):
        if delete_existing:
            logger.info("Deleting folder %s", folderpath)
            shutil.rmtree(folderpath)
    else:
        logger.info("Creating folder %s", folderpath)
        os.makedirs(folderpath)


def save_data(dataAll):

    if SETTINGS['SAVING_DATA'] == 'PICKLE':
        save_pickle(dataAll)

    elif SETTINGS['SAVING_DATA'] == 'CSV':
        save_csv(dataAll)

    else:
        raise ValueError("Saving method is not valid")


def load_data():

    if SETTINGS['SAVING_DATA'] == 'PICKLE':
        return load_pickle()

    elif SETTINGS['SAVING_DATA'] == 'CSV':
        return load_csv()

    else:
        raise ValueError("Saving method is not valid")


def save_pickle(dataAll):
    saving_path = os.path.join(
        SETTINGS['PATH_OUTPUT'], 'dataAll.pickle')

    logger.info("=========================================")
    logger.info("Saving data to %s", saving_path)

    pickle.dump(dataAll, open(saving_path, 'wb'), protocol=2)


def load_pickle():
    saving_path = os.path.join(
        SETTINGS['PATH_OUTPUT'], 'dataAll.pickle')

    logger.info("=========================================")
    logger.info("Loading data from %s", saving_path)

    dataAll = pickle.load(
        open(os.path.join(SETTINGS['PATH_OUTPUT'], 'dataAll.pickle'), 'rb'))

    return dataAll


def save_csv(dataAll):
    folderpath = SETTINGS['PATH_OUTPUT']
    logger.info("=========================================")
    logger.info("Saving data to %s", folderpath)

    for subj in SETTINGS['SUBJECT_LIST']:
        foldersubj = join(folderpath, "{}".format(subj))
        create_folder(foldersubj)

        dataSubj = dataAll[subj]

        dataSubj['data'].to_csv(join(foldersubj, 'data.csv'),index=False)
        dataSubj['groundtruth'].to_csv(join(foldersubj, 'groundtruth.csv'),index=False)
        dataSubj['feature'].to_csv(join(foldersubj, 'feature.csv'),index=False)
        dataSubj['segmentation_in_time'].to_csv(
            join(foldersubj, 'segmentation_in_time.csv'),index=False)
        dataSubj['segmentation_in_index'].to_csv(
            join(foldersubj, 'segmentation_in_index.csv'),index=False)

        np.savetxt(join(foldersubj, 'label_segmentation.csv'),
                   dataSubj['label_segmentation'], fmt='%d')


def load_csv():
    folderpath = SETTINGS['PATH_OUTPUT']
    logger.info("=========================================")
    logger.info("Loading data from %s", folderpath)

    dataAll = {}

    for subj in SETTINGS['SUBJECT_LIST']:
        foldersubj = os.path.join(folderpath, "{}".format(subj))

        dataSubj = {}
        dataSubj['data'] = pd.read_csv(
            join(foldersubj, 'data.csv'), index_col=False)
        dataSubj['groundtruth'] = pd.read_csv(
            join(foldersubj, 'groundtruth.csv'), index_col=False)
        dataSubj['feature'] = pd.read_csv(
            join(foldersubj, 'feature.csv'), index_col=False)
        dataSubj['segmentation_in_time'] = pd.read_csv(
            join(foldersubj, 'segmentation_in_time.csv'), index_col=False)
        dataSubj['segmentation_in_index'] = pd.read_csv(
            join(foldersubj, 'segmentation_in_index.csv'), index_col=False)

        dataSubj['label_segmentation'] = np.genfromtxt(
            join(foldersubj, 'label_segmentation.csv')).astype(int)

        dataAll[subj] = dataSubj

    return dataAll
