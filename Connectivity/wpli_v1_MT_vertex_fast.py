#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:37:57 2020

@author: a_shishkina
"""
import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
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

all_v1_mt = []

for subject in SUBJECTS:

    allepochs = mne.read_epochs(savepath + subject + '/' + subject + '_isi-epo.fif')
    # Sort epochs according to experimental conditions in the post-stimulus interval
    #slow_epo_isi = allepochs.__getitem__('V1')
    #fast_epo_isi = allepochs.__getitem__('V3')
    
    fname_inv = savepath + subject + '/' + subject + '_inv'
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    src = inverse_operator['src'] 
    
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2   
    
    # Source Estimates
    stcs_fast = apply_inverse_epochs(allepochs, inverse_operator, lambda2, method='sLORETA',
                                     pick_ori="normal")
    # Read labels
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_rh.label')
    
    # Extract Source Estimates from labels
    stcs_fast_v1 = apply_inverse_epochs(allepochs, inverse_operator, lambda2, method='sLORETA',
                                     pick_ori="normal", label=v1_label)
    stcs_fast_mt = apply_inverse_epochs(allepochs, inverse_operator, lambda2, method='sLORETA',
                                     pick_ori="normal", label=mt_label)
    
    # Find index of vertices of interest in original stc
    idx_v1 = np.searchsorted(stcs_fast[0].vertices[1], stcs_fast_v1[0].vertices[1])
    idx_mt = np.searchsorted(stcs_fast[0].vertices[1], stcs_fast_mt[0].vertices[1])

    # Construct indices to estimate connectivity between the label time course
    indices = seed_target_indices([idx_v1], [idx_mt])
        
    # Compute the WPLI2_debiased in the frequency range 
    fmin = 2.
    fmax = 40.
    sfreq = allepochs.info['sfreq']  # the sampling frequency
    
    wpli2, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        stcs_fast, method='wpli2_debiased', sfreq=sfreq, indices=indices,
        mode='fourier', fmin=fmin, fmax=fmax, n_jobs=1)
    # Average over values between all vertices [v1_vertices * mt_vertices]
    wpli2_avg = wpli2.mean(0)        
    all_v1_mt.append(wpli2_avg)
stop = timeit.default_timer()
time = stop-start

# Save   
np.save(savepath + 'wpli2_debiased/' + 'all_v1_mt_vertices_fast_allepo', all_v1_mt)
all_subj = np.load(savepath + 'wpli2_debiased/' + 'all_v1_mt_vertices_fast_allepo.npy')
avg_subj = all_subj.mean(0)
plt.plot(freqs, avg_subj)
plt.title('wpli2_debiased between V1 and MT averaged over all subjects')
plt.plot(np.arange(42), np.zeros(42), 'r--')
plt.ylabel('wpli2_debiased')
plt.xlabel('frequency')
plt.savefig(savepath + 'wpli2_debiased/' + 'all_v1_mt_vertices_fast_allepo')
