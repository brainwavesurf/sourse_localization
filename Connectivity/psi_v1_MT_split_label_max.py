#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# divide labels on sublabels and compute PSI between vertices with max amplitude within sublabels
# split V1 on 16 parts and MT on 5 parts
"""
Created on Fri Jul 24 17:36:08 2020

@author: a_shishkina
"""

import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator, source_band_induced_power
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
    
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_rh.label')
    
    v1_label_part = v1_label.split(parts=16, subject='Case'+subject, subjects_dir=subjects_dir, freesurfer=True)
    mt_label_part = mt_label.split(parts=5, subject='Case'+subject, subjects_dir=subjects_dir, freesurfer=True)
    
    psi_slow_v1_mt_all = np.zeros([len(v1_label_part),1])
    slow_v1_pow = []
    
    for label_num_v1 in np.arange(len(v1_label_part)):
        
        stcs_slow_v1_induced = source_band_induced_power(slow_epo_isi, inverse_operator, dict(alpha_beta=[8,17]), method='sLORETA', n_cycles=2, 
                                                         label=v1_label_part[label_num_v1])
        #sort array of vertices in descending order according power
        pow_idx_v1 = stcs_slow_v1_induced['alpha_beta'].data.mean(1).argsort()
        vert_descend_v1 = stcs_slow_v1_induced['alpha_beta'].vertices[1][pow_idx_v1[::-1]]
        #take only 20 vertices with max power
        v1_label.vertices = vert_descend_v1[0]
        
        psi_slow_v1_mt = np.zeros([len(mt_label_part),1])
        
        for label_num_mt in np.arange(len(mt_label_part)):
        
            stcs_slow_mt_induced = source_band_induced_power(slow_epo_isi, inverse_operator, dict(alpha_beta=[8,17]), method='sLORETA', n_cycles=2, 
                                                             label=mt_label_part[label_num_mt])
            
            #same for mt
            pow_idx_mt = stcs_slow_mt_induced['alpha_beta'].data.mean(1).argsort()
            vert_descend_mt = stcs_slow_mt_induced['alpha_beta'].vertices[1][pow_idx_mt[::-1]]
            #take only 20 vertices with max power
            mt_label.vertices=vert_descend_mt[0]
            
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
            
            psi_slow_v1_mt[label_num_mt] = np.array(psi_slow[0][0])
            avg_mt = np.average(psi_slow_v1_mt, axis=0)
        psi_slow_v1_mt_all[label_num_v1] = avg_mt
        avg_v1 = np.nanmean(psi_slow_v1_mt_all, axis=0)
        stop = timeit.default_timer()
    avg_v1_mt.append(avg_v1)
stop = timeit.default_timer()
time = stop-start
np.save(savepath + 'psi/' + 'all_v1_mt_16_5_max_pow_rh', avg_v1_mt)

all_max = np.load(savepath + 'psi/' + 'all_v1_mt_16_5_max_pow_rh.npy')
plt.scatter(np.arange(42), all_max.mean(1), c='green')
plt.title('Max power sublabels time courses')
plt.plot(np.arange(42), np.zeros(42), 'r--')
plt.ylim(-0.3,0.3)
plt.ylabel('PSI')
plt.xlabel('Subjects')
plt.savefig(savepath + 'psi/' + 'split_max_sublabels_v1_mt_16_5')

