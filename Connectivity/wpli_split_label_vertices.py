#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# divide labels on sublabels and compute wpli2_debiased between all vertices in one V1 sublable [5] and one MT sublabel [1]
# 7-8 vertices
"""
Created on Thu Aug 13 20:25:20 2020

@author: a_shishkina
"""
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
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
    
    # Read labels for V1 and MT 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_rh.label')
    
    v1_label_part = v1_label.split(parts=16, subject='Case'+subject, subjects_dir=subjects_dir, freesurfer=True)
    mt_label_part = mt_label.split(parts=4, subject='Case'+subject, subjects_dir=subjects_dir, freesurfer=True)
    
    fast_v1_pow = []
    
    #one
    stcs_fast_v1 = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                        pick_ori="normal", label=v1_label_part[5])
    psi_fast_v1_mt_all = np.zeros([len(stcs_fast_v1[0].vertices[1]), 31])
    
    for vert_num_v1 in range(len(stcs_fast_v1[0].vertices[1])):
        
        # Now, we generate seed time series from each vertex in the left V1
        v1_label_part[5].vertices = stcs_fast_v1[0].vertices[1][vert_num_v1]
            
        seed_ts_fast_v1 = mne.extract_label_time_course(stcs_fast_v1, v1_label_part[5], src, mode='mean_flip',
                                                            verbose='error')
        
        #two
        stcs_fast_mt = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                        pick_ori="normal", label=mt_label_part[1])
        psi_fast_v1_mt = np.zeros([len(stcs_fast_mt[0].vertices[1]),31])
        
        for vert_num_mt in range(len(stcs_fast_mt[0].vertices[1])):
            
            # Now, we generate seed time series from each vertex in the left V1
            mt_label_part[2].vertices = stcs_fast_mt[0].vertices[1][vert_num_mt]
                
            seed_ts_fast_mt = mne.extract_label_time_course(stcs_fast_mt, mt_label_part[2], src, mode='mean_flip',
                                                                verbose='error')
            
            # Combine the seed time course with the source estimates. 
            comb_ts_fast = list(zip(seed_ts_fast_v1, seed_ts_fast_mt))
        
            # Construct indices to estimate connectivity between the label time course
            # and all source space time courses
            vertices = [src[i]['vertno'] for i in range(2)]
        
            indices = (np.array([0]), np.array([1]))
        
            # Compute the PSI in the frequency range 11Hz-17Hz.
            fmin = 2.
            fmax = 40.
            sfreq = fast_epo_isi.info['sfreq']  # the sampling frequency
            
            wpli2, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                comb_ts_fast, method='wpli2_debiased', sfreq=sfreq, indices=indices,
                mode='fourier', fmin=fmin, fmax=fmax, n_jobs=1)
            
            psi_fast_v1_mt[vert_num_mt] = wpli2
            avg_mt = psi_fast_v1_mt.mean(0)
        psi_fast_v1_mt_all[vert_num_v1] = avg_mt
        avg_v1 = psi_fast_v1_mt_all.mean(0)    
    all_v1_mt.append(avg_v1)
stop = timeit.default_timer()
time = stop-start

# Save   
np.save(savepath + 'wpli2_debiased/' + 'all_split_v1_mt_5_1_vertices', all_v1_mt)
all_subj = np.load(savepath + 'wpli2_debiased/' + 'all_split_v1_mt_5_1_vertices.npy')
avg_subj = all_subj.mean(0)
plt.plot(freqs, avg_subj)
plt.title('wpli2_debiased between V1 and MT averaged over all subjects')
plt.plot(np.arange(42), np.zeros(42), 'r--')
plt.ylabel('wpli2_debiased')
plt.xlabel('frequency')
plt.savefig(savepath + 'wpli2_debiased/' + 'split_sublabel_v1_mt_5_1')

