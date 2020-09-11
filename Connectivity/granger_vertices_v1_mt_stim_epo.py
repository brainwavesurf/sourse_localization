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

all_subj_v1_mt_asd_slow = np.empty((1,203,len(SUBJ_ASD)))
all_subj_v1_mt_asd_med = np.empty((1,203,len(SUBJ_ASD)))
all_subj_v1_mt_asd_fast = np.empty((1,203,len(SUBJ_ASD)))

all_subj_mt_v1_asd_slow = np.empty((1,203,len(SUBJ_ASD)))
all_subj_mt_v1_asd_med = np.empty((1,203,len(SUBJ_ASD)))
all_subj_mt_v1_asd_fast = np.empty((1,203,len(SUBJ_ASD)))

all_subj_v1_mt_nt_slow = np.empty((1,203,len(SUBJ_NT)))
all_subj_v1_mt_nt_med = np.empty((1,203,len(SUBJ_NT)))
all_subj_v1_mt_nt_fast = np.empty((1,203,len(SUBJ_NT)))

all_subj_mt_v1_nt_slow = np.empty((1,203,len(SUBJ_NT)))
all_subj_mt_v1_nt_med = np.empty((1,203,len(SUBJ_NT)))
all_subj_mt_v1_nt_fast = np.empty((1,203,len(SUBJ_NT)))

idx_subj_asd = []
idx_subj_nt = []
for subject in SUBJECTS:
    if subject in SUBJ_ASD:
        idx_subj_asd.append(SUBJECTS.index(subject))
    else:
        idx_subj_nt.append(SUBJECTS.index(subject))
        
    allepochs = mne.read_epochs(savepath + subject + '/' + subject + '_stim-epo.fif')
    # Sort epochs according to experimental conditions in the post-stimulus interval
    slow_epo = allepochs.__getitem__('V1')
    medium_epo = allepochs.__getitem__('V2')
    fast_epo = allepochs.__getitem__('V3')
    idx_order = np.zeros((len(allepochs.events),3), dtype='int32')
    idx_order[0:len(slow_epo.events)] = slow_epo.events
    idx_order[len(slow_epo.events):len(slow_epo.events)+len(medium_epo.events)] = medium_epo.events
    idx_order[len(slow_epo.events)+len(medium_epo.events):len(allepochs.events)] = fast_epo.events

    fname_inv = savepath + subject + '/' + subject + '_inv'
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    src = inverse_operator['src'] 
    
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    
    # Read labels for V1 and MT 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_rh.label')

    stcs_v1 = apply_inverse_epochs(allepochs, inverse_operator, lambda2, method='sLORETA',
                                   pick_ori="normal", label=v1_label)
    stcs_mt = apply_inverse_epochs(allepochs, inverse_operator, lambda2, method='sLORETA',
                                   pick_ori="normal", label=mt_label)
    
    src = inverse_operator['src']  # the source space used
    stc_label_v1 = mne.stc_to_label(stcs_v1[0], src=src, subjects_dir=subjects_dir, smooth=False)
    stc_label_mt = mne.stc_to_label(stcs_mt[0], src=src, subjects_dir=subjects_dir, smooth=False)
    
    vertices_v1 = range(len(stc_label_v1[1].vertices))
    vertices_mt = range(len(stc_label_mt[1].vertices))
    
    tcs_v1 = []
    for vert_num_v1 in vertices_v1:
        # Now, we generate seed time series from each vertex in the left V1
        vertex_v1 = mne.label.select_sources('Case'+subject, label=stc_label_v1[1], location=vert_num_v1, 
                                             subjects_dir=subjects_dir)
        seed_tc_v1 = mne.extract_label_time_course(stcs_v1, vertex_v1, src, mode='mean_flip', verbose='error')
        tcs_v1.append(seed_tc_v1)
     
    tcs_mt = []
    for vert_num_mt in vertices_mt:
        # Now, we generate seed time series from each vertex in the left V1
        vertex_mt = mne.label.select_sources('Case'+subject, label=stc_label_mt[1], location=vert_num_mt, 
                                             subjects_dir=subjects_dir)
        seed_ts_mt = mne.extract_label_time_course(stcs_mt, vertex_mt, src, mode='mean_flip', verbose='error')
        tcs_mt.append(seed_ts_mt)
     
    # Create signals input
    datarray = np.asarray(tcs_v1)
    signal_v1 = np.transpose(datarray.mean(2),(2,1,0)) #(times,epochs,signals))
    datarray = np.asarray(tcs_mt)
    signal_mt = np.transpose(datarray.mean(2),(2,1,0)) 
    
    sfreq = allepochs.info['sfreq']
    nvert_v1 = len(stc_label_v1[1].vertices)
    nvert_mt = len(stc_label_mt[1].vertices)
    
    idx_slow = len(slow_epo.events)
    idx_med = len(medium_epo.events)
    idx_all = len(allepochs.events)
    
    # Granger causality from v1 to mt
    granger_v1_mt_avg_slow = np.empty((1,203,nvert_v1))
    granger_v1_mt_avg_med = np.empty((1,203,nvert_v1))
    granger_v1_mt_avg_fast = np.empty((1,203,nvert_v1))

    granger_mt_v1_avg_slow = np.empty((1,203,nvert_v1))
    granger_mt_v1_avg_med = np.empty((1,203,nvert_v1))
    granger_mt_v1_avg_fast = np.empty((1,203,nvert_v1))
    
    for idx_v1 in range(nvert_v1):
        granger_v1_mt_slow = np.empty((1,203,nvert_mt))
        granger_v1_mt_med = np.empty((1,203,nvert_mt))
        granger_v1_mt_fast = np.empty((1,203,nvert_mt))

        granger_mt_v1_slow = np.empty((1,203,nvert_mt))
        granger_mt_v1_med = np.empty((1,203,nvert_mt))
        granger_mt_v1_fast = np.empty((1,203,nvert_mt))

        for idx_mt in range(nvert_mt):
            # from v1 to mt
            signal_v1_mt_slow = np.append(signal_v1[:,0:idx_slow,idx_v1,np.newaxis], 
                                          signal_mt[:,0:idx_slow,idx_mt,np.newaxis], axis=2)
            signal_v1_mt_med = np.append(signal_v1[:,idx_slow:idx_slow+idx_med,idx_v1,np.newaxis], 
                                         signal_mt[:,idx_slow:idx_slow+idx_med,idx_mt,np.newaxis], axis=2)
            signal_v1_mt_fast = np.append(signal_v1[:,idx_slow+idx_med:idx_all,idx_v1,np.newaxis], 
                                          signal_mt[:,idx_slow+idx_med:idx_all,idx_mt,np.newaxis], axis=2)
            # from mt to v1
            signal_mt_v1_slow = np.append(signal_mt[:,0:idx_slow,idx_mt,np.newaxis], 
                                          signal_v1[:,0:idx_slow,idx_v1,np.newaxis], axis=2)
            signal_mt_v1_med = np.append(signal_mt[:,idx_slow:idx_slow+idx_med,idx_mt,np.newaxis], 
                                         signal_v1[:,idx_slow:idx_slow+idx_med,idx_v1,np.newaxis], axis=2)
            signal_mt_v1_fast = np.append(signal_mt[:,idx_slow+idx_med:idx_all,idx_mt,np.newaxis], 
                                          signal_v1[:,idx_slow+idx_med:idx_all,idx_v1,np.newaxis], axis=2)
            
            # Compute granger causality
            m_v1_mt_slow = Multitaper(signal_v1_mt_slow, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            m_v1_mt_med = Multitaper(signal_v1_mt_med, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            m_v1_mt_fast = Multitaper(signal_v1_mt_fast, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            
            m_mt_v1_slow = Multitaper(signal_mt_v1_slow, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            m_mt_v1_med = Multitaper(signal_mt_v1_med, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            m_mt_v1_fast = Multitaper(signal_mt_v1_fast, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            
            c_v1_mt_slow = Connectivity(fourier_coefficients=m_v1_mt_slow.fft(), frequencies=m_v1_mt_slow.frequencies)
            c_v1_mt_med = Connectivity(fourier_coefficients=m_v1_mt_med.fft(), frequencies=m_v1_mt_med.frequencies)
            c_v1_mt_fast = Connectivity(fourier_coefficients=m_v1_mt_fast.fft(), frequencies=m_v1_mt_fast.frequencies)
            
            c_mt_v1_slow = Connectivity(fourier_coefficients=m_mt_v1_slow.fft(), frequencies=m_mt_v1_slow.frequencies)
            c_mt_v1_med = Connectivity(fourier_coefficients=m_mt_v1_med.fft(), frequencies=m_mt_v1_med.frequencies)
            c_mt_v1_fast = Connectivity(fourier_coefficients=m_mt_v1_fast.fft(), frequencies=m_mt_v1_fast.frequencies)
            
            granger_v1_mt_slow[:,:,idx_mt] = c_v1_mt_slow.pairwise_spectral_granger_prediction()[...,0,1]
            granger_v1_mt_med[:,:,idx_mt] = c_v1_mt_med.pairwise_spectral_granger_prediction()[...,0,1]
            granger_v1_mt_fast[:,:,idx_mt] = c_v1_mt_fast.pairwise_spectral_granger_prediction()[...,0,1]

            granger_mt_v1_slow[:,:,idx_mt] = c_mt_v1_slow.pairwise_spectral_granger_prediction()[...,0,1]
            granger_mt_v1_med[:,:,idx_mt] = c_mt_v1_med.pairwise_spectral_granger_prediction()[...,0,1]
            granger_mt_v1_fast[:,:,idx_mt] = c_mt_v1_fast.pairwise_spectral_granger_prediction()[...,0,1]
        
        granger_v1_mt_avg_slow[:,:,idx_v1] = granger_v1_mt_slow.mean(2)
        granger_v1_mt_avg_med[:,:,idx_v1] = granger_v1_mt_med.mean(2)
        granger_v1_mt_avg_fast[:,:,idx_v1] = granger_v1_mt_fast.mean(2)
        
        granger_mt_v1_avg_slow[:,:,idx_v1] = granger_mt_v1_slow.mean(2)
        granger_mt_v1_avg_med[:,:,idx_v1] = granger_mt_v1_med.mean(2)
        granger_mt_v1_avg_fast[:,:,idx_v1] = granger_mt_v1_fast.mean(2)

    granger_all_v1_mt_slow = granger_v1_mt_avg_slow.mean(2)
    granger_all_v1_mt_med = granger_v1_mt_avg_med.mean(2)
    granger_all_v1_mt_fast = granger_v1_mt_avg_fast.mean(2)

    granger_all_mt_v1_slow = granger_mt_v1_avg_slow.mean(2)
    granger_all_mt_v1_med = granger_mt_v1_avg_med.mean(2)
    granger_all_mt_v1_fast = granger_mt_v1_avg_fast.mean(2)

    # Sort by group
    if subject in SUBJ_ASD:
        idx_subj_asd = SUBJ_ASD.index(subject)
        all_subj_v1_mt_asd_slow[:,:,idx_subj_asd] = granger_all_v1_mt_slow
        all_subj_v1_mt_asd_med[:,:,idx_subj_asd] = granger_all_v1_mt_med
        all_subj_v1_mt_asd_fast[:,:,idx_subj_asd] = granger_all_v1_mt_fast
        
        all_subj_mt_v1_asd_slow[:,:,idx_subj_asd] = granger_all_mt_v1_slow
        all_subj_mt_v1_asd_med[:,:,idx_subj_asd] = granger_all_mt_v1_med
        all_subj_mt_v1_asd_fast[:,:,idx_subj_asd] = granger_all_mt_v1_fast
        
    else:
        idx_subj_nt = SUBJ_NT.index(subject)
        all_subj_v1_mt_nt_slow[:,:,idx_subj_nt] = granger_all_v1_mt_slow
        all_subj_v1_mt_nt_med[:,:,idx_subj_nt] = granger_all_v1_mt_med
        all_subj_v1_mt_nt_fast[:,:,idx_subj_nt] = granger_all_v1_mt_fast
        
        all_subj_mt_v1_nt_slow[:,:,idx_subj_nt] = granger_all_mt_v1_slow
        all_subj_mt_v1_nt_med[:,:,idx_subj_nt] = granger_all_mt_v1_med
        all_subj_mt_v1_nt_fast[:,:,idx_subj_nt] = granger_all_mt_v1_fast
    
np.save(savepath + 'granger/' + 'all_subj_v1_mt_asd_slow', all_subj_v1_mt_asd_slow)
np.save(savepath + 'granger/' + 'all_subj_v1_mt_asd_med', all_subj_v1_mt_asd_med)
np.save(savepath + 'granger/' + 'all_subj_v1_mt_asd_fast', all_subj_v1_mt_asd_fast)

np.save(savepath + 'granger/' + 'all_subj_mt_v1_asd_slow', all_subj_mt_v1_asd_slow)
np.save(savepath + 'granger/' + 'all_subj_mt_v1_asd_med', all_subj_mt_v1_asd_med)
np.save(savepath + 'granger/' + 'all_subj_mt_v1_asd_fast', all_subj_mt_v1_asd_fast)

np.save(savepath + 'granger/' + 'all_subj_v1_mt_nt_slow', all_subj_v1_mt_nt_slow)
np.save(savepath + 'granger/' + 'all_subj_v1_mt_nt_med', all_subj_v1_mt_nt_med)
np.save(savepath + 'granger/' + 'all_subj_v1_mt_nt_fast', all_subj_v1_mt_nt_fast)

np.save(savepath + 'granger/' + 'all_subj_mt_v1_nt_slow', all_subj_mt_v1_nt_slow)
np.save(savepath + 'granger/' + 'all_subj_mt_v1_nt_med', all_subj_mt_v1_nt_med)
np.save(savepath + 'granger/' + 'all_subj_mt_v1_nt_fast', all_subj_mt_v1_nt_fast)

stop = timeit.default_timer()
time = stop - start

# Plot
plt.plot(c_v1_mt_slow.frequencies, all_subj_v1_mt_nt_slow.mean(2)[0,:], label='v1->mt')
plt.plot(c_v1_mt_slow.frequencies, all_subj_mt_v1_nt_slow.mean(2)[0,:], label='mt->v1')
plt.legend()
plt.xlim(2,70)
plt.title('nt_slow')
plt.ylabel('Granger Causaliy Value')
plt.xlabel('frequency')
plt.savefig(savepath + 'granger/' + 'all_subj_v1_mt_nt_slow')

# Compute the Directed Assimetry Index
dai_nt_slow = (all_subj_v1_mt_nt_slow - all_subj_mt_v1_nt_slow)/(all_subj_v1_mt_nt_slow + all_subj_mt_v1_nt_slow)
dai_asd_slow = (all_subj_v1_mt_asd_slow - all_subj_mt_v1_asd_slow)/(all_subj_v1_mt_asd_slow + all_subj_mt_v1_asd_slow)

dai_nt_med = (all_subj_v1_mt_nt_med - all_subj_mt_v1_nt_med)/(all_subj_v1_mt_nt_med + all_subj_mt_v1_nt_med)
dai_asd_med = (all_subj_v1_mt_asd_med - all_subj_mt_v1_asd_med)/(all_subj_v1_mt_asd_med + all_subj_mt_v1_asd_med)

dai_nt_fast = (all_subj_v1_mt_nt_fast - all_subj_mt_v1_nt_fast)/(all_subj_v1_mt_nt_fast + all_subj_mt_v1_nt_fast)
dai_asd_fast = (all_subj_v1_mt_asd_fast - all_subj_mt_v1_asd_fast)/(all_subj_v1_mt_asd_fast + all_subj_mt_v1_asd_fast)

# Plot
plt.plot(c_v1_mt_slow.frequencies, dai_nt_med.mean(2)[0,:], label='nt_med')
plt.plot(c_v1_mt_slow.frequencies, dai_asd_med.mean(2)[0,:], label='asd_med')
plt.plot(c_v1_mt_slow.frequencies, np.zeros(len(c_v1_mt_slow.frequencies)), 'k--')
plt.legend()
plt.xlim(2,70)
plt.title('Directed Asymmetry Index')
plt.ylabel('DAI')
plt.xlabel('frequency')
plt.savefig(savepath + 'granger/' + 'v1_mt_dai_med')