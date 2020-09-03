#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:33:33 2020

@author: a_shishkina
"""

import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

from spectral_connectivity import Multitaper, Connectivity
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

all_subj_v1_v4_asd = []
all_subj_v4_v1_asd = []
all_subj_v1_v4_nt = []
all_subj_v4_v1_nt = []

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
    
    # Read labels for V1 and v4 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    v4_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V4_rh.label')

    stcs_v1 = apply_inverse_epochs(allepochs, inverse_operator, lambda2, method='sLORETA',
                                   pick_ori="normal", label=v1_label)
    stcs_v4 = apply_inverse_epochs(allepochs, inverse_operator, lambda2, method='sLORETA',
                                   pick_ori="normal", label=v4_label)
    
    src = inverse_operator['src']  # the source space used
    stc_label_v1 = mne.stc_to_label(stcs_v1[0], src=src, subjects_dir=subjects_dir, smooth=False)
    stc_label_v4 = mne.stc_to_label(stcs_v4[0], src=src, subjects_dir=subjects_dir, smooth=False)
    
    vertices_v1 = range(len(stc_label_v1[1].vertices))
    vertices_v4 = range(len(stc_label_v4[1].vertices))
    
    tcs_v1 = []
    tcs_v4 = []
    
    for vert_num_v1 in vertices_v1:
        
        #one
        # Now, we generate seed time series from each vertex in the left V1
        vertex_v1 = mne.label.select_sources('Case'+subject, label=stc_label_v1[1], location=vert_num_v1, 
                                             subjects_dir=subjects_dir)
    
        seed_tc_v1 = mne.extract_label_time_course(stcs_v1, vertex_v1, src, mode='mean_flip',
                                                   verbose='error')
        tcs_v1.append(seed_tc_v1)
        
    for vert_num_v4 in vertices_v4:
            
        #two
        # Now, we generate seed time series from each vertex in the left V1
        vertex_v4 = mne.label.select_sources('Case'+subject, label=stc_label_v4[1], location=vert_num_v4, 
                                             subjects_dir=subjects_dir)
        
        seed_ts_v4 = mne.extract_label_time_course(stcs_v4, vertex_v4, src, mode='mean_flip',
                                                   verbose='error')
        tcs_v4.append(seed_ts_v4)
    
    sfreq = allepochs.info['sfreq'] 
    
    # Create signals input
    datarray = np.asarray(tcs_v1)
    signal_v1 = np.transpose(datarray.mean(2),(2,1,0)) #(times,epochs,signals))
    datarray = np.asarray(tcs_v4)
    signal_v4 = np.transpose(datarray.mean(2),(2,1,0)) 
    
    # Granger causality from v1 to v4
    granger_v1_v4 = np.empty((1,203,0))
    for idx_v1 in range(len(stc_label_v1[1].vertices)):
        granger_v4 = np.empty((1,203,0))
        for idx_v4 in range(len(stc_label_v4[1].vertices)):
            signal = np.append(signal_v1[:,:,idx_v1,np.newaxis], signal_v4[:,:,idx_v4,np.newaxis], axis=2)
            # Compute granger causality
            m = Multitaper(signal, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            c = Connectivity(fourier_coefficients=m.fft(), frequencies=m.frequencies)
            granger = c.pairwise_spectral_granger_prediction()
            granger_v4 = np.append(granger_v4, granger[...,0,1,np.newaxis], axis=2)
        granger_v1_v4 = np.append(granger_v1_v4, granger_v4.mean(2)[...,np.newaxis], axis=2)
    granger_all_v1_v4 = granger_v1_v4.mean(2)
    
    
    # Granger causality from v4 to v1
    granger_v1_v4 = np.empty((1,203,0))
    for idx_v1 in range(len(stc_label_v1[1].vertices)):
        granger_v4 = np.empty((1,203,0))
        for idx_v4 in range(len(stc_label_v4[1].vertices)):
            signal = np.append(signal_v4[:,:,idx_v4,np.newaxis], signal_v1[:,:,idx_v1,np.newaxis], axis=2)
            # Compute granger causality
            m = Multitaper(signal, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            c = Connectivity(fourier_coefficients=m.fft(), frequencies=m.frequencies)
            granger = c.pairwise_spectral_granger_prediction()
            granger_v4 = np.append(granger_v4, granger[...,0,1,np.newaxis], axis=2)
        granger_v1_v4 = np.append(granger_v1_v4, granger_v4.mean(2)[...,np.newaxis], axis=2)
    granger_all_v4_v1 = granger_v1_v4.mean(2)
    
    if subject in SUBJ_ASD:
        all_subj_v1_v4_asd.append(granger_all_v1_v4)
        all_subj_v4_v1_asd.append(granger_all_v4_v1)
    else:
        all_subj_v1_v4_nt.append(granger_all_v1_v4)
        all_subj_v4_v1_nt.append(granger_all_v4_v1)
    
np.save(savepath + 'granger/' + 'all_v1_v4_nt_vert', all_subj_v1_v4_nt)
np.save(savepath + 'granger/' + 'all_v4_v1_nt_vert', all_subj_v4_v1_nt)
np.save(savepath + 'granger/' + 'all_v1_v4_asd_vert', all_subj_v1_v4_asd)
np.save(savepath + 'granger/' + 'all_v4_v1_asd_vert', all_subj_v4_v1_asd)

stop = timeit.default_timer()
time = stop - start

# Load granger causality values for all subjects
all_subj_v1_v4_asd = np.load(savepath + 'granger/' + 'all_v1_v4_asd_vert.npy')
all_subj_v1_v4_nt = np.load(savepath + 'granger/' + 'all_v1_v4_nt_vert.npy')
all_subj_v4_v1_asd = np.load(savepath + 'granger/' + 'all_v4_v1_asd_vert.npy')
all_subj_v4_v1_nt = np.load(savepath + 'granger/' + 'all_v4_v1_nt_vert.npy')

# Average over subjects
avg_subj_v1_v4_asd = all_subj_v1_v4_asd.mean(0)
avg_subj_v1_v4_nt = all_subj_v1_v4_nt.mean(0)
avg_subj_v4_v1_asd = all_subj_v4_v1_asd.mean(0)
avg_subj_v4_v1_nt = all_subj_v4_v1_nt.mean(0)


plt.plot(c.frequencies, avg_subj_v1_v4_asd[0,:], label='v1->v4')
plt.plot(c.frequencies, avg_subj_v4_v1_asd[0,:], label='v4->v1')
plt.legend()
plt.ylim(0,0.05)
plt.xlim(2,70)
plt.title('asd')
plt.ylabel('Granger Causaliy Value')
plt.xlabel('frequency')
plt.savefig(savepath + 'granger/' + 'all_v1_v4_asd_vert')

# Compute the Directed Assimetry Index
dai_nt = (avg_subj_v1_v4_nt - avg_subj_v4_v1_nt)/(avg_subj_v1_v4_nt + avg_subj_v4_v1_nt)
dai_asd = (avg_subj_v1_v4_asd - avg_subj_v4_v1_asd)/(avg_subj_v1_v4_asd + avg_subj_v4_v1_asd)

# Plot
plt.plot(c.frequencies, dai_nt[0,:], label='nt')
plt.plot(c.frequencies, dai_asd[0,:], label='asd')
plt.plot(c.frequencies, np.zeros(len(c.frequencies)), 'k--')
plt.legend()
plt.xlim(2,40)
plt.title('Directed Asymmetry Index')
plt.ylabel('DAI')
plt.xlabel('frequency')
plt.savefig(savepath + 'granger/' + 'v1_v4_dai')
