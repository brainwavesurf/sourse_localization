#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# debiased estimator of squared wheighted phase lag index computation 
# between time courses averaged in V1 and V4 labels

"""
Created on Tue Jul 22 14:37:57 2020

@author: a_shishkina
"""
import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity


import matplotlib.pyplot as plt
import numpy as np
import timeit

start = timeit.default_timer()

datapath = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
savepath = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/Results_Alpha_and_Gamma/'
subjects_dir = datapath + 'freesurfersubjects'

#load subj info
SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383'] 

SUBJECTS = SUBJ_ASD + SUBJ_NT

avg_v1_v4 = []

for subject in SUBJECTS:
 
    allepochs = mne.read_epochs(savepath + subject + '/' + subject + '_isi-epo.fif')
    # Sort epochs according to experimental conditions in the post-stimulus interval
    #slow_epo_isi = allepochs.__getitem__('V1')
    fast_epo_isi = allepochs.__getitem__('V3')
    
    fname_inv = savepath + subject + '/' + subject + '_inv'
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    
    snr = 3.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    
    # Read labels for V1 and MT 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    v4_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V4_rh.label')

    # Compute inverse solution for each epochs 
    stcs_fast = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                        pick_ori="normal")
    
    # Extract time courses from vertices
    src = inverse_operator['src']  # the source space used
    seed_ts_fast_v1 = mne.extract_label_time_course(stcs_fast, v1_label, src, mode='mean_flip', verbose='error')
    seed_ts_fast_v4 = mne.extract_label_time_course(stcs_fast, v4_label, src, mode='mean_flip', verbose='error')
    
    # Combine two time courses 
    comb_ts_fast = list(zip(seed_ts_fast_v1, seed_ts_fast_v4))
    
    # Create indices for two label time courses
    indices = (np.array([0]), np.array([1]))
    
    # Compute the PSI in the frequency range 8Hz-17Hz.
    fmin = 2.
    fmax = 40.
    sfreq = fast_epo_isi.info['sfreq']  # the sampling frequency
    
    wpli2, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        comb_ts_fast, method='wpli2_debiased', sfreq=sfreq, indices=indices,
        mode='fourier', fmin=fmin, fmax=fmax, n_jobs=1)
    
    # Add wpli2 values for each subject
    avg_v1_v4.append(wpli2)
    
# Save   
np.save(savepath + 'wpli2_debiased/' + 'all_v1_v4_avg_freq', avg_v1_v4)

plt.plot(freqs, avg_v1_v4[0][0])
plt.title('wpli2_debiased between V1 and V4 averaged over all subjects')
plt.plot(np.arange(42), np.zeros(42), 'r--')
plt.ylabel('wpli2_debiased')
plt.xlabel('frequency')
plt.savefig(savepath + 'wpli2_debiased/' + 'all_v1_v4_avg')