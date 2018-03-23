import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from PASDAC.settings import SETTINGS
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import logging
from sklearn.svm import LinearSVC
logger = logging.getLogger(__name__)
def fillinMatrix(smallArr, colIndicesArr, nCols):
    """
    In case prediction does not have contain all classes
    """
    
    nRows = smallArr.shape[0]
    fullArr = np.zeros([nRows, nCols])

    for i in range(colIndicesArr.size):
        fullArr[:,colIndicesArr[i]] = smallArr[:,i]

    return fullArr


def classification(trainData, trainLabels, testData):
    """Train model with trainData and trainLabels, then predict testLabels given testData.
    Output one hot representation and probability

    Parameters
    ----------
        trainingData:               dataFrame
        trainLabels:                dataFrame
        testData:                   dataFrame

    Return
    ------
        result:                     dataFrame
        probaDf:                    dataFrame

    """
    method = SETTINGS['CLASSIFIER']
    nClass = len(SETTINGS['CLASS_LABELS'])
    classLabels = SETTINGS['CLASS_LABELS']

    trainLabelsUnqArr = np.unique(trainLabels)

    if method == 'NaiveBayes':
        classifier = GaussianNB()
        model = classifier.fit(trainData, trainLabels)
        result = model.predict(testData)
        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
    elif method == 'knnVoting':

        classifier = KNeighborsClassifier(5)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
        
    elif method == 'RandomForests':
        
        classifier = RandomForestClassifier(max_depth=10, random_state=0)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
        ############################################
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(trainData.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(trainData.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(trainData.shape[1]), indices)
        plt.xlim([-1, trainData.shape[1]])
        plt.show()
        
    elif method == 'SVM':
        
        classifier = svm.SVC(C = 3, gamma = 0.003, probability = True)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
        
    elif method == 'AdaBoost':
        
        classifier = AdaBoostClassifier()
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
                ############################################
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(trainData.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(trainData.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(trainData.shape[1]), indices)
        plt.xlim([-1, trainData.shape[1]])
        plt.show()
        
    elif method == 'NeuralNetwork':
        classifier = MLPClassifier(alpha=1)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
        
    elif method == 'LogisticRegression':
        classifier = LogisticRegression()
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
        
    elif method == 'LinearSVM':
        classifier = LinearSVC(random_state=0)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)
        
        
                ############################################
        importances = model.coef_
      #  std = np.std([tree.feature_importances_ for tree in model.estimators_],
        plt.plot(importances.shape[1])
        plt.ylabel('some numbers')
        plt.show()
        
        logger.info(model.coef_)
#        proba = model.predict_proba(testData)
#        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
#        probaDf = pd.DataFrame(data=proba, columns=classLabels)

        
    logger.info(method)   

    return result, probaDf