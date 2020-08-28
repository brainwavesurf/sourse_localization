#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:23:01 2020

@author: a_shishkina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import packages
import mne
from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs
import matplotlib.pyplot as plt
import timeit
import numpy as np

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
    
for subject in SUBJECTS:
    
    allepochs = mne.read_epochs(savepath + subject + '/' + subject + '_isi-epo.fif')
    # Sort epochs according to experimental conditions in the post-stimulus interval
    slow_epo_isi = allepochs.__getitem__('V1')
    fast_epo_isi = allepochs.__getitem__('V3')
    
    fname_inv = savepath + subject + '/' + subject + '_inv'
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    src = inverse_operator['src'] 
    
    # Read labels for V1 and MT 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_rh.label')

    #stc calculation    
    method = "sLORETA"
    snr = 3.
    #lambda2 = 0.05
    lambda2 = 1. / snr ** 2
    bandwidth = 'hann' 
    
    #for slow interstimulus epochs in v1
    n_epochs_use = slow_epo_isi.events.shape[0]
    stcs_slow_isi_v1 = compute_source_psd_epochs(slow_epo_isi[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth, label=v1_label,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_slow_isi_v1 in enumerate(stcs_slow_isi_v1):
        psd_avg += stc_slow_isi_v1.data
    psd_avg /= n_epochs_use
    freqs = stc_slow_isi_v1.times  
    stc_slow_isi_v1.data = psd_avg  

    #for fast interstimulus epochs in v1
    n_epochs_use = fast_epo_isi.events.shape[0]
    stcs_fast_isi_v1 = compute_source_psd_epochs(fast_epo_isi[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth, label=v1_label,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_fast_isi_v1 in enumerate(stcs_fast_isi_v1):
        psd_avg += stc_fast_isi_v1.data
    psd_avg /= n_epochs_use
    freqs = stc_fast_isi_v1.times  
    stc_fast_isi_v1.data = psd_avg
    
    #for slow interstimulus epochs in mt
    n_epochs_use = slow_epo_isi.events.shape[0]
    stcs_slow_isi_mt = compute_source_psd_epochs(slow_epo_isi[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth, label=mt_label,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_slow_isi_mt in enumerate(stcs_slow_isi_mt):
        psd_avg += stc_slow_isi_mt.data
    psd_avg /= n_epochs_use
    freqs = stc_slow_isi_mt.times  
    stc_slow_isi_mt.data = psd_avg
    
    #for fast interstimulus epochs in mt
    n_epochs_use = fast_epo_isi.events.shape[0]
    stcs_fast_isi_mt = compute_source_psd_epochs(fast_epo_isi[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth, label=mt_label,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_fast_isi_mt in enumerate(stcs_fast_isi_mt):
        psd_avg += stc_fast_isi_mt.data
    psd_avg /= n_epochs_use
    freqs = stc_fast_isi_mt.times  
    stc_fast_isi_mt.data = psd_avg
    
    # Freqs and Power
    fast_v1 = np.array([stc_fast_isi_v1.times, stc_fast_isi_v1.data.mean(0)])
    slow_v1 = np.array([stc_slow_isi_v1.times, stc_slow_isi_v1.data.mean(0)])
    fast_mt = np.array([stc_fast_isi_mt.times, stc_fast_isi_mt.data.mean(0)])
    slow_mt = np.array([stc_slow_isi_mt.times, stc_slow_isi_mt.data.mean(0)])
    
    allepo_v1_mt = np.array([fast_v1, slow_v1, fast_mt, slow_mt])
    
    all_subj.append(allepo_v1_mt)  
# Save   
np.save(savepath + 'pow/' + 'all_fast_slow_v1_mt_freq_pow', all_subj)
all_subj = np.load(savepath + 'pow/' + 'all_fast_slow_v1_mt_freq_pow.npy')
avg_subj = all_subj.mean(0)

# Plot
plt.plot(avg_subj[0,0,:], avg_subj[0,1,:], label='fast_v1')
plt.plot(avg_subj[1,0,:], avg_subj[1,1,:], label='slow_v1')
plt.title('Spectral power in MT averaged over all subjects')
plt.ylabel('Power')
plt.xlabel('Frequency')
plt.legend()

plt.plot(avg_subj[2,0,:], avg_subj[2,1,:], label='fast_mt')
plt.plot(avg_subj[3,0,:], avg_subj[3,1,:], label='slow_mt')
plt.legend()

plt.savefig(savepath + 'pow/' + 'all_fast_slow_mt_freq_pow')

        
        