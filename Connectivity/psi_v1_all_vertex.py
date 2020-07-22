#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:37:57 2020

@author: a_shishkina
"""
import mne
from mne.minimum_norm import apply_inverse_epochs
from mne.connectivity import seed_target_indices, phase_slope_index
from mne import io

import numpy as np
import scipy.io

datapath = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
savepath = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/Results_Alpha_and_Gamma/'
subjects_dir = datapath + 'freesurfersubjects'

subject = '0102'
 
fname_raw = datapath + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw.fif'
fname_inv = savepath + subject + '/' + subject + '_inv.fif'

# Load Data
raw = io.Raw(fname_raw, preload=True)
raw.filter(2, 40)
inverse_operator = mne.read_inverse_operator(fname_inv, verbose=None)
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

# Resample epochs
if allepochs.info['sfreq']>500:
    allepochs.resample(500)
    
# Sort epochs according to experimental conditions in the post-stimulus interval
allepochs.crop(tmin=-0.8, tmax=0)
slow_epo_isi = allepochs.__getitem__('V1')
fast_epo_isi = allepochs.__getitem__('V3')

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2   

stcs_slow = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                 pick_ori="normal", return_generator=True)
stcs_fast = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                 pick_ori="normal", return_generator=True)
    
label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_lh.label')
vertices = range(len(label.get_vertices_used())-1)
for vert_num in vertices:
    
    stcs_slow = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                pick_ori="normal", return_generator=True)
    stcs_fast = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                                pick_ori="normal", return_generator=True)
    # Now, we generate seed time series from each vertex in the left V1
    src = inverse_operator['src']  # the source space used

    vertex = mne.label.select_sources('Case0102', label=label, location=vert_num, subjects_dir=subjects_dir)
    
    seed_ts_slow = mne.extract_label_time_course(stcs_slow, vertex, src, mode='mean_flip',
                                                 allow_empty=True, return_generator=True)
    seed_ts_fast = mne.extract_label_time_course(stcs_fast, vertex, src, mode='mean_flip', 
                                                 allow_empty=True, return_generator=True)
    
    # Combine the seed time course with the source estimates. 
    comb_ts_slow = list(zip(seed_ts_slow, stcs_slow))
    comb_ts_fast = list(zip(seed_ts_fast, stcs_fast))

    # Construct indices to estimate connectivity between the label time course
    # and all source space time courses
    vertices = [src[i]['vertno'] for i in range(2)]
    n_signals_tot = 1 + len(vertices[0]) + len(vertices[1])

    indices = seed_target_indices([0], np.arange(1, n_signals_tot))

    # Compute the PSI in the frequency range 11Hz-17Hz.
    fmin = 11.
    fmax = 17.
    sfreq = slow_epo_isi.info['sfreq']  # the sampling frequency
    
    psi_slow, freqs, times, n_epochs, _ = phase_slope_index(
        comb_ts_slow, mode='fourier', indices=indices, sfreq=sfreq,
        fmin=fmin, fmax=fmax)

    psi_fast, freqs, times, n_epochs, _ = phase_slope_index(
        comb_ts_fast, mode='fourier', indices=indices, sfreq=sfreq,
        fmin=fmin, fmax=fmax)

    # Generate a SourceEstimate with the PSI. This is simple since we used a single
    # seed (inspect the indices variable to see how the PSI scores are arranged in
    # the output)
    psi_slow_stc = mne.SourceEstimate(psi_slow, vertices=vertices, tmin=-0.8, tstep=1)
    psi_fast_stc = mne.SourceEstimate(psi_fast, vertices=vertices, tmin=-0.8, tstep=1)

psi_slow_stc.save(savepath + subject + '/' + subject + '_psi_slow_isi')
psi_fast_stc.save(savepath + subject + '/' + subject + '_psi_fast_isi')