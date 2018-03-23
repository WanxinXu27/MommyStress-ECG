import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal
from PASDAC.Preprocessing.smoothing import smooth_boxcar

def test_smooth_boxcar():
	data = pd.DataFrame(data=np.zeros(20)+1, columns=["Sensor1"], dtype = float)
	smoothed = smooth_boxcar(data, ["Sensor1"], 30)

	assert_array_almost_equal(data['Sensor1'].as_matrix(), smoothed['Sensor1'].as_matrix())