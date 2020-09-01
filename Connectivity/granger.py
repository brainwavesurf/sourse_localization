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

all_subj = []

for subject in SUBJ_ASD:
 
    allepochs = mne.read_epochs(savepath + subject + '/' + subject + '_isi-epo.fif')
    # Sort epochs according to experimental conditions in the post-stimulus interval
    #slow_epo_isi = allepochs.__getitem__('V1')
    #fast_epo_isi = allepochs.__getitem__('V3')
    
    fname_inv = savepath + subject + '/' + subject + '_inv'
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    src = inverse_operator['src'] 
    
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    
    # Read labels for V1 and MT 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    #v4_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V4_rh.label')
    mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_rh.label')

    # Compute inverse solution for each epochs 
    stcs_fast = apply_inverse_epochs(allepochs, inverse_operator, lambda2, method='sLORETA',
                                     pick_ori="normal")
    
    # Extract time courses from vertices
    seed_ts_fast_v1 = mne.extract_label_time_course(stcs_fast, v1_label, src, mode='mean_flip', verbose='error')
    #seed_ts_fast_v4 = mne.extract_label_time_course(stcs_fast, v4_label, src, mode='mean_flip', verbose='error')
    seed_ts_fast_mt = mne.extract_label_time_course(stcs_fast, mt_label, src, mode='mean_flip', verbose='error')

    comb_ts_fast = list(zip(seed_ts_fast_v1, seed_ts_fast_mt))
    sfreq = allepochs.info['sfreq'] 
    
    # Create signals input
    datarray = np.asarray(comb_ts_fast)
    signal = np.transpose(datarray.mean(2),(2,0,1)) #(401,78,2))
    
    # Compute granger causality
    m = Multitaper(signal, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
    c = Connectivity(fourier_coefficients=m.fft(), frequencies=m.frequencies)
    granger = c.pairwise_spectral_granger_prediction()
    
    all_subj.append(granger)
np.save(savepath + 'granger/' + 'all_v1_mt_avg_new_package_asd', all_subj)

# Load granger causality values for all subjects
all_subj_v1_mt_asd = np.load(savepath + 'granger/' + 'all_v1_mt_avg_new_package_asd.npy')
all_subj_v1_mt_nt = np.load(savepath + 'granger/' + 'all_v1_mt_avg_new_package_nt.npy')
all_subj_mt_v1_asd = np.load(savepath + 'granger/' + 'all_mt_v1_avg_new_package_asd.npy')
all_subj_mt_v1_nt = np.load(savepath + 'granger/' + 'all_mt_v1_avg_new_package_nt.npy')

# Average over subjects
avg_subj_v1_mt_asd = all_subj_v1_mt_asd.mean(0)
avg_subj_v1_mt_nt = all_subj_v1_mt_nt.mean(0)
avg_subj_mt_v1_asd = all_subj_mt_v1_asd.mean(0)
avg_subj_mt_v1_nt = all_subj_mt_v1_nt.mean(0)

# Plot
plt.plot(c.frequencies, avg_subj_v1_mt_nt[...,0,1].squeeze(), label='v1->mt')
plt.plot(c.frequencies, avg_subj_mt_v1_nt[...,0,1].squeeze(), label='mt->v1')
plt.legend()
plt.ylim(0,0.05)
plt.xlim(2,70)
plt.title('asd')
plt.ylabel('Granger Causaliy Value')
plt.xlabel('frequency')
plt.savefig(savepath + 'granger/' + 'all_v1_mt_avg_new_package_asd')


# all_subj_v1_v4_nt_fast = np.load(savepath + 'granger/' + 'all_v1_v4_avg_new_package_nt_fast.npy')
# all_subj_v1_v4_asd_fast = np.load(savepath + 'granger/' + 'all_v1_v4_avg_new_package_asd_fast.npy')
# all_subj_v4_v1_nt_fast = np.load(savepath + 'granger/' + 'all_v4_v1_avg_new_package_nt_fast.npy')
# all_subj_v4_v1_asd_fast = np.load(savepath + 'granger/' + 'all_v4_v1_avg_new_package_asd_fast.npy')

# avg_subj_v1_v4_nt_fast = all_subj_v1_v4_nt_fast.mean(0)
# avg_subj_v1_v4_asd_fast = all_subj_v1_v4_asd_fast.mean(0)
# avg_subj_v4_v1_nt_fast = all_subj_v4_v1_nt_fast.mean(0)
# avg_subj_v4_v1_asd_fast = all_subj_v4_v1_asd_fast.mean(0)
    
