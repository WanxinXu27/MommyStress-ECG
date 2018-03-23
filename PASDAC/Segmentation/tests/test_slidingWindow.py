from PASDAC.Segmentation import segment
import numpy as np
import pandas as pd


def test_slidingWindow():

	dataDf = pd.DataFrame({'Time':np.arange(0, 100000, 50)})

	segmentDf = segment(dataDf)

