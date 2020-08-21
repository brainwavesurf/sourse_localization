#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# debiased estimator of squared wheighted phase lag index computation 
# between time courses averaged in V1 and MT labels

"""
Created on Tue Jul 22 14:37:57 2020

@author: a_shishkina
"""
import mne
from mne.minimum_norm import apply_inverse, read_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity, seed_target_indices


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

avg_v1_mt = []

for subject in SUBJECTS:
 
    allepochs = mne.read_epochs(savepath + subject + '/' + subject + '_isi-epo.fif')
    # Sort epochs according to experimental conditions in the post-stimulus interval
    slow_epo_isi = allepochs.__getitem__('V1')
    fast_epo_isi = allepochs.__getitem__('V3')
    
    fname_inv = savepath + subject + '/' + subject + '_inv'
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    
    snr = 3.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    
    # Read labels for V1 and MT 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_rh.label')
    
    # Compute inverse solution for each epochs 
    stcs_slow = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                        pick_ori="normal")
    
    # Extract time courses from vertices
    src = inverse_operator['src']  # the source space used
    seed_ts_slow_v1 = mne.extract_label_time_course(stcs_slow_v1, v1_label, src, mode='mean_flip', verbose='error')
    seed_ts_slow_mt = mne.extract_label_time_course(stcs_slow_mt, mt_label, src, mode='mean_flip', verbose='error')
    
    # Combine two time courses 
    comb_ts_slow = list(zip(seed_ts_slow_v1, seed_ts_slow_mt))
    
    # Create indices for two label time courses
    indices = (np.array([0]), np.array([1]))
    
    # Compute the PSI in the frequency range 8Hz-17Hz.
    fmin = 8.
    fmax = 17.
    sfreq = slow_epo_isi.info['sfreq']  # the sampling frequency
    
    wpli2, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        comb_ts_slow, method='wpli2_debiased', sfreq=sfreq, indices=indices,
        mode='fourier', faverage=True, fmin=fmin, fmax=fmax, n_jobs=1)
    
    # Add wpli2 values for each subject
    avg_v1_mt.append(wpli2)
    
# Save   
np.save(savepath + 'wpli2_debiased/' + 'all_v1_mt_avg_rh_fast', avg_v1_mt)
stop = timeit.default_timer()   
time = stop - start 


# Alternative way

    # Compute inverse solution for averaged epochs 
    slow_epo_isi_avg = slow_epo_isi.average()
    stc_slow = apply_inverse(slow_epo_isi_avg, inverse_operator, lambda2, method='sLORETA',
                             pick_ori="normal")
    # Extract stc within specific label
    stc_slow_v1 = stc_slow.in_label(v1_label)
    stc_slow_mt = stc_slow.in_label(mt_label)
    
    # Find number and index of vertices
    seed_vertno_v1 = stc_slow_v1.vertices[1]
    seed_vertno_mt = stc_slow_mt.vertices[1]
    
    seed_idx_v1 = np.searchsorted(stc_slow_v1.vertices[1], seed_vertno_v1)
    seed_idx_mt = np.searchsorted(stc_slow_mt.vertices[1], seed_vertno_mt)
    
    indices = seed_target_indices([seed_idx_mt],[seed_idx_v1])
    
    # Compute inverse solution for each epochs 
    stcs_slow = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                        pick_ori="normal")
  
    # Compute the PSI in the frequency range 8Hz-17Hz.
    fmin = 8.
    fmax = 17.
    sfreq = slow_epo_isi.info['sfreq']  # the sampling frequency
    
    wpli2, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        stcs_slow, method='wpli2_debiased', sfreq=sfreq, indices=indices,
        mode='fourier', faverage=True, fmin=fmin, fmax=fmax, n_jobs=1)
