# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:20:39 2018

@author: Administrator
"""
import os
import logging
import numpy as np
import pandas as pd
from biosppy.signals import ecg, tools

logger = logging.getLogger(__name__)


def calculateECGFeatures(dataFrame, smooth=True, normalize=True):
    '''
Inputs:     dataFrame   pandas.dataFrame containing clip of raw ECG data, should have two columns of
                        Timestamps (ms) and Sample (V)
                        
            smooth      optional Boolean, default value is True, flag for whether to smooth input raw
                        ECG data or not
            
            normalize   option Boolean, default value is True, flag for whether to normalize input raw
                        ECG data or not

Outputs:    features    pandas.dataFrame for input clip of data, each column corresponds to a different
                        feature, included features are as follows:
                        - mean                      - ultra low frequency power
                        - median                    - very low frequency power
                        - max                       - low frequency power
                        - variance                  - high frequency power
                        - standard deviation        - LF/HF ratio
                        - absolute deviation        - total power
                        - kurtois                   - R-R interval
                        - skew
'''

    # ----------------------------------------------------------------------------------------------------
    # Data Preprocessing ---------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------

    
    time = dataFrame['Timestamp (ms)'].values
    data = dataFrame['Sample (V)'].values

    ## Smooth raw ECG data
    if smooth:
        smooth_signal = tools.smoother(data, kernel='median', size=5) # Convolutional 5x5 kernel window
        data = smooth_signal['signal']
        
    ## Normalize raw ECG data
    if normalize:
        norm_signal = tools.normalize(data)
        data = norm_signal['signal']
        
        
       
        
    # ----------------------------------------------------------------------------------------------------        
    # Begin Features Extraction Code ---------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    ## Calculate basic statistic features
    s_mean, s_med, s_max, s_var, s_std_dev, s_abs_dev, s_kurtois, s_skew = tools.signal_stats(data)
    
    ## Obtain Power Spectra
    power_freqs, power_spectrum = tools.power_spectrum(data, sampling_rate=2.0, decibel = False)
    
    ## Calculate Ultra Low-Frequency Power (ULF Band: <= 0.003 Hz)
#    power_ulf = tools.band_power(freqs=power_freqs, power=power_spectrum, frequency=[0, 0.003])
#    power_ulf = np.absolute(power_ulf['avg_power'])
    
    # Calculate Very Low-Frequency Power (VLF Band: 0.0033 - 0.04 Hz)
    power_vlf = tools.band_power(freqs=power_freqs, power=power_spectrum, frequency=[0.0033, 0.04])
    power_vlf = np.absolute(power_vlf['avg_power'])
    
    # Calculate Low-Frequency Power (LF Band: 0.04 - 0.15 Hz Hz)
    power_lf = tools.band_power(freqs=power_freqs, power=power_spectrum, frequency=[0.04, 0.15])
    power_lf = np.absolute(power_lf['avg_power'])
    
    ## Calculate High-Frequency Power (HF Band: 0.15 - 0.40 Hz)
    power_hf = tools.band_power(freqs=power_freqs, power=power_spectrum, frequency=[0.15, 0.40])
    power_hf = np.absolute(power_hf['avg_power'])
    
    ## Calculate LF/HF Ratio
    lf_hf_ratio = power_lf / power_hf
    
    ## Calculate Total Power (VLF + LF + HF power)
    power_total = power_vlf + power_lf + power_hf
    
    # Calculate R peak indices
    r_peaks = ecg.engzee_segmenter(data, sampling_rate = 500.0)
    if np.size(r_peaks) != 1:
        rr_indices = time[r_peaks]
        rr_intervals = np.mean(np.diff(rr_indices))
    else:
        rr_intervals = np.nan # NaN indicates not enough R peaks within window to calculate R-R interval
    
    # ----------------------------------------------------------------------------------------------------
    # Collect Features and convert into dataFrame --------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    
    signal_features = {'mean': s_mean,
                       'median': s_med,
                       'max': s_max,
                       'variance': s_var,
                       'std_dev': s_std_dev,
                       'abs_dev': s_abs_dev,
                       'kurtois': s_kurtois,
                       'skew': s_skew,
#                       'ulf_power': power_ulf,
                       'vlf_power': power_vlf,
                       'lf_power': power_lf,
                       'hf_power': power_hf,
                       'lf_hf_ratio': lf_hf_ratio,
                       'rr_interval': rr_intervals,
}
    
    features = pd.Series(signal_features)
    
    a = [ s_mean, s_med, s_max, s_var, s_std_dev, s_abs_dev, s_kurtois, s_skew,
#         power_ulf,
             power_vlf,power_lf,power_hf,lf_hf_ratio,
             rr_intervals
   #      rr_indices
         ]
    return a


if __name__ == "__main__":

    # Enter path to .csv file containing raw ECG data
    path = r'E:\NU2018winter\MommyStress-ECG\Data2R'

    # load raw ECG signal
    df = pd.read_csv(os.path.join(path, 'subject2_ecg_data.csv'))
    features = calculateECGFeatures(df[0:2000])
    print (features)




