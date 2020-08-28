#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:37:57 2020

@author: a_shishkina
"""
import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
from mne.connectivity import phase_slope_index, seed_target_indices


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
    fast_epo_isi = allepochs.__getitem__('V3')
    
    fname_inv = savepath + subject + '/' + subject + '_inv'
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    src = inverse_operator['src'] 
    
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2   
    
    # Source Estimates
    stcs = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                     pick_ori="normal")
    # Read labels
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_rh.label')
    
    # Extract Source Estimates from labels
    stcs_v1 = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                     pick_ori="normal", label=v1_label)
    stcs_mt = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                     pick_ori="normal", label=mt_label)
    
    # Find index of vertices of interest in original stc
    idx_v1 = np.searchsorted(stcs[0].vertices[1], stcs_v1[0].vertices[1])
    idx_mt = np.searchsorted(stcs[0].vertices[1], stcs_mt[0].vertices[1])

    # Construct indices to estimate connectivity between the label time course
    indices = seed_target_indices([idx_v1], [idx_mt])
        
    # Compute the WPLI2_debiased in the frequency range 
    fmin = 8.
    fmax = 15.
    sfreq = fast_epo_isi.info['sfreq']  # the sampling frequency
    
    # Compute the PSI in the frequency range 8Hz-15Hz.  
    psi, freqs, times, n_epochs, _ = phase_slope_index(
        stcs, mode='fourier', sfreq=sfreq, indices=indices,
        fmin=fmin, fmax=fmax, n_jobs=1)
    # Average over values between all vertices [v1_vertices * mt_vertices]
    psi_avg = psi.mean(0)        
    all_v1_mt.append(psi_avg)
stop = timeit.default_timer()
time = stop-start

# Save   
np.save(savepath + 'psi/' + 'all_v1_mt_vertices_fast_fastepo', all_v1_mt)
all_subj = np.load(savepath + 'psi/' + 'all_v1_mt_vertices_fast_fastepo.npy')
plt.scatter(np.arange(42), all_subj[:,0])
plt.plot(np.arange(42), np.zeros(42), 'r--')
plt.title('PSI between V1 and MT for all subjects')
plt.ylabel('PSI')
plt.xlabel('subjects')
plt.savefig(savepath + 'psi/' + 'all_v1_mt_vertices_fast_fastepo')