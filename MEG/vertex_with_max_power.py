#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:33:33 2020

@author: a_shishkina

Define a vertex in V1 with maximum gamma power
"""

import mne
from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs


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

SUBJECTS_good_pow = ['0101','0102','0104','0105','0106','0107']

for subject in SUBJECTS_good_pow:
 
    allepochs = mne.read_epochs(savepath + subject + '/' + subject + '_stim-epo_gamma.fif')
    pre_allepochs = mne.read_epochs(savepath + subject + '/' + subject + '_prestim-epo_gamma.fif')
    # Sort epochs according to experimental conditions in the post-stimulus interval
    slow_epo_stim = allepochs.__getitem__('V1')
    slow_epo_prestim = pre_allepochs.__getitem__('V1')
    
    fname_inv = savepath + subject + '/' + subject + '_inv'
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    src = inverse_operator['src'] 
    
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    
    # Read labels for V1 and MT 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    
    #for V1   
    method = "sLORETA"
    snr = 3.
    #lambda2 = 0.05
    lambda2 = 1. / snr ** 2
    bandwidth = 4.0
    
    #for prestim epochs
    n_epochs_use = slow_epo_prestim.events.shape[0]
    stcs_slow_prestim = compute_source_psd_epochs(slow_epo_prestim[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=45, fmax=75,
                                 bandwidth=bandwidth, label=v1_label,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_slow_prestim in enumerate(stcs_slow_prestim):
        psd_avg += stc_slow_prestim.data
    psd_avg /= n_epochs_use
    freqs = stc_slow_prestim.times  
    stc_slow_prestim.data = psd_avg  

    
    #for stim epoch
    n_epochs_use = slow_epo_stim.events.shape[0]
    stcs_slow_stim = compute_source_psd_epochs(slow_epo_stim[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=45, fmax=75,
                                 bandwidth=bandwidth, label=v1_label,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_slow_stim in enumerate(stcs_slow_stim):
        psd_avg += stc_slow_stim.data
    psd_avg /= n_epochs_use
    freqs = stc_slow_stim.times  
    stc_slow_stim.data = psd_avg 
    
    max_idx_v1 = np.argmax((stc_slow_stim.data - stc_slow_prestim.data)/100, axis=0) #124
    
    max_idx_v1_avg = np.argmax((stc_slow_stim.data.mean(1) - stc_slow_prestim.data.mean(1))/100, axis=0) #124
    