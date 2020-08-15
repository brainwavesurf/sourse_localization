#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#phase slope index computation between time courses averages in V1 and MT labels

"""
Created on Tue Jul 22 14:37:57 2020

@author: a_shishkina
"""
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import phase_slope_index


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
    src = inverse_operator['src'] 
    
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    
    # Read labels for V1 and MT 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_rh.label')
    
    # Compute inverse solution for each epochs 
    stcs_slow_v1 = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                        pick_ori="normal", return_generator=True)
    stcs_slow_mt = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                     pick_ori="normal", return_generator=True)
    
    #extract time courses from vertices
    src = inverse_operator['src']  # the source space used
    seed_ts_slow_v1 = mne.extract_label_time_course(stcs_slow_v1, v1_label, src, mode='mean_flip', verbose='error')
    seed_ts_slow_mt = mne.extract_label_time_course(stcs_slow_mt, mt_label, src, mode='mean_flip', verbose='error')
    
    comb_ts_slow = list(zip(seed_ts_slow_v1, seed_ts_slow_mt))

    indices = (np.array([0]), np.array([1]))
    
    # Compute the PSI in the frequency range 8Hz-17Hz.
    fmin = 8.
    fmax = 17.
    sfreq = slow_epo_isi.info['sfreq']  # the sampling frequency
    
    psi_slow, freqs, times, n_epochs, _ = phase_slope_index(
        comb_ts_slow, mode='fourier', sfreq=sfreq, indices=indices,
        fmin=fmin, fmax=fmax)
    avg_v1_mt.append(psi_slow)
np.save(savepath + 'psi/' + 'all_v1_mt_avg_rh', avg_v1_mt)
stop = timeit.default_timer()   
time = stop - start 

all_avg = np.load(savepath + 'psi/' + 'all_v1_mt_avg_rh.npy')
plt.scatter(np.arange(42), all_avg.mean(1))
plt.title('V1 MT labels avg time courses')
plt.plot(np.arange(42), np.zeros(42), 'r--')
plt.ylim(-0.3,0.3)
plt.ylabel('PSI')
plt.xlabel('Subjects')
plt.savefig(savepath + 'psi/' + 'v1_mt_avg')

