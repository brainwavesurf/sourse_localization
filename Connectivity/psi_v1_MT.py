#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:37:57 2020

@author: a_shishkina
"""
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import phase_slope_index
from mne import io

import numpy as np
import scipy.io

datapath = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
savepath = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/Results_Alpha_and_Gamma/'
subjects_dir = datapath + 'freesurfersubjects'

subject = '0102'
 
fname_raw = datapath + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw.fif'
fname_inv = savepath + subject + '/' + subject + '_inv'

# Load Data
raw = io.Raw(fname_raw, preload=True)
raw.filter(2, 40)
inverse_operator = read_inverse_operator(fname_inv, verbose=None)
events = mne.find_events(raw, stim_channel='STI101', verbose=True, shortest_event=1)

# Make epochs from raw    

delay = 8 
events[:,0] = events[:,0]+delay/1000.*raw.info['sfreq']
ev = np.sort(np.concatenate((np.where(events[:,2]==2), np.where(events[:,2]==4), np.where(events[:,2]==8)), axis=1))  
relevantevents = events[ev,:][0]

# Extract epochs from raw
events_id = dict(V1=2, V2=4, V3=8)
presim_sec = -1.
poststim_sec = 1.4

# Load info about preceding events
info_mat = scipy.io.loadmat(datapath + 'Results_Alpha_and_Gamma/'+ subject + '/' + subject + '_info.mat')
good_epo = info_mat['ALLINFO']['ep_order_num'][0][0][0]-1
info_file = info_mat['ALLINFO']['stim_type_previous_tr'][0][0][0]
goodevents = relevantevents[good_epo]
goodevents[:,2] = info_file

# Define epochs 
allepochs = mne.Epochs(raw, goodevents, events_id, tmin=presim_sec, tmax=poststim_sec, baseline=(None, 0), proj=False, preload=True)
allepochs.save(savepath + subject + '/' + subject + '_')
# Resample epochs
if allepochs.info['sfreq']>500:
    allepochs.resample(500)
    
# Sort epochs according to experimental conditions in the post-stimulus interval
allepochs.crop(tmin=-0.8, tmax=0)
slow_epo_isi = allepochs.__getitem__('V1')
fast_epo_isi = allepochs.__getitem__('V3')

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2   

v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_lh.label')
mt_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_MT_lh.label')

stcs_slow_v1 = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                 pick_ori="normal", label=v1_label)
stcs_slow_mt = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                 pick_ori="normal", label=mt_label)

src = inverse_operator['src']  # the source space used
stc_label_v1 = mne.stc_to_label(stcs_slow_v1[0], src=src, subjects_dir=subjects_dir, smooth=False)
stc_label_mt = mne.stc_to_label(stcs_slow_mt[0], src=src, subjects_dir=subjects_dir, smooth=False)

vertices_v1 = range(len(stc_label_v1[0].vertices))
vertices_mt = range(len(stc_label_mt[0].vertices))

for vert_num_v1 in vertices_v1:
    
    #one
    stcs_slow = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                pick_ori="normal", return_generator=True)

    # Now, we generate seed time series from each vertex in the left V1
    vertex_v1 = mne.label.select_sources('Case0102', label=stc_label_v1[0], location=vert_num_v1, 
                                      subjects_dir=subjects_dir)
    
    seed_ts_slow_v1 = mne.extract_label_time_course(stcs_slow, vertex_v1, src, mode='mean_flip',
                                                verbose='error')
    
    psi_slow_v1_mt = []
    for vert_num_mt in vertices_mt:
        
        stcs_slow = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                    pick_ori="normal", return_generator=False)
    
        # Now, we generate seed time series from each vertex in the left V1
        vertex_mt = mne.label.select_sources('Case0102', label=stc_label_mt[0], location=vert_num_mt, 
                                          subjects_dir=subjects_dir)
        
        seed_ts_slow_mt = mne.extract_label_time_course(stcs_slow, vertex_mt, src, mode='mean_flip',
                                                    verbose='error')
        
        # Combine the seed time course with the source estimates. 
        comb_ts_slow = list(zip(seed_ts_slow_v1, seed_ts_slow_mt))
    
        # Construct indices to estimate connectivity between the label time course
        # and all source space time courses
        vertices = [src[i]['vertno'] for i in range(2)]
    
        indices = (np.array([0]), np.array([1]))
    
        # Compute the PSI in the frequency range 11Hz-17Hz.
        fmin = 11.
        fmax = 17.
        sfreq = slow_epo_isi.info['sfreq']  # the sampling frequency
        
        psi_slow, freqs, times, n_epochs, _ = phase_slope_index(
            comb_ts_slow, mode='fourier', sfreq=sfreq, indices=indices,
            fmin=fmin, fmax=fmax)
        psi_slow_v1_mt.append(psi_slow)
    
