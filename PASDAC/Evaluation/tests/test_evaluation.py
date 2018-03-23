from PASDAC.Evaluation.metrics import coreEvaluate
import numpy as np


def test_evaluation():

    labelsArr = np.array([1, 1, 1, 2, 2, 2])
    predArr = np.array([1, 1, 2, 2, 1, 1])

    print(coreEvaluate(labelsArr, predArr, 1))