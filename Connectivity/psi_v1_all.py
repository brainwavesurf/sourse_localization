#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:37:57 2020

@author: a_shishkina
"""

import numpy as np

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import seed_target_indices, phase_slope_index
from mne import io
import scipy.io

datapath = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
mypath = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/'
subjects_dir = datapath + 'freesurfersubjects'
subject = '0102'
 
subjpath = datapath  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
savepath = mypath + 'Results_Alpha_and_Gamma/'

raw_fname = datapath + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw.fif'
#load forward model
fwd = mne.read_forward_solution(savepath + subject + '/' + subject + '_fwd', verbose=None)
 
#make epochs from raw    
raw = io.Raw(raw_fname, preload=True)
raw.filter(2, 40)
#events
events= mne.find_events(raw, stim_channel='STI101', verbose=True, shortest_event=1)
delay = 8 
events[:,0] = events[:,0]+delay/1000.*raw.info['sfreq']
ev = np.sort(np.concatenate((np.where(events[:,2]==2), np.where(events[:,2]==4), np.where(events[:,2]==8)), axis=1))  
relevantevents = events[ev,:][0]
#extract epochs from raw
events_id = dict(V1=2, V2=4, V3=8)
presim_sec = -1.
poststim_sec = 1.4
#load info about preceding events
info_mat = scipy.io.loadmat(datapath + 'Results_Alpha_and_Gamma/'+ subject + '/' + subject + '_info.mat')
good_epo = info_mat['ALLINFO']['ep_order_num'][0][0][0]-1
info_file = info_mat['ALLINFO']['stim_type_previous_tr'][0][0][0]
goodevents = relevantevents[good_epo]
goodevents[:,2] = info_file
allepochs = mne.Epochs(raw, goodevents, events_id, tmin=presim_sec, tmax=poststim_sec, baseline=(None, 0), proj=False, preload=True)

if allepochs.info['sfreq']>500:
    allepochs.resample(500)
    
#interstimulus epochs
allepochs.crop(tmin=-0.8, tmax=0)
slow_epo_isi = allepochs.__getitem__('V1')
fast_epo_isi = allepochs.__getitem__('V3')

#load noise covariance matrix from empty room data
noise_cov = mne.read_cov(savepath + subject + '/' + subject + 'noise_cov_2_40Hz', verbose=None)

#inverse operator
inverse_operator = make_inverse_operator(allepochs.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
stcs_slow = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                            pick_ori="normal", return_generator=True)
stcs_fast = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                            pick_ori="normal", return_generator=True)
# Now, we generate seed time series by averaging the activity in the left V1

label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_lh.label')
src = inverse_operator['src']  # the source space used
seed_ts_slow = mne.extract_label_time_course(stcs_slow, label, src, mode='mean_flip', verbose='error')
seed_ts_fast = mne.extract_label_time_course(stcs_fast, label, src, mode='mean_flip', verbose='error')

# Combine the seed time course with the source estimates. 
stcs_slow = apply_inverse_epochs(slow_epo_isi, inverse_operator, lambda2, method='sLORETA',
                            pick_ori="normal", return_generator=True)
stcs_fast = apply_inverse_epochs(fast_epo_isi, inverse_operator, lambda2, method='sLORETA',
                            pick_ori="normal", return_generator=True)
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
    comb_ts_slow, mode='multitaper', indices=indices, sfreq=sfreq,
    fmin=fmin, fmax=fmax)

psi_fast, freqs, times, n_epochs, _ = phase_slope_index(
    comb_ts_fast, mode='multitaper', indices=indices, sfreq=sfreq,
    fmin=fmin, fmax=fmax)

# Generate a SourceEstimate with the PSI. This is simple since we used a single
# seed (inspect the indices variable to see how the PSI scores are arranged in
# the output)
psi_slow_stc = mne.SourceEstimate(psi_slow, vertices=vertices, tmin=-0.8, tstep=1)
psi_fast_stc = mne.SourceEstimate(psi_fast, vertices=vertices, tmin=-0.8, tstep=1)

psi_slow_stc.save(savepath + subject + '/' + subject + '_psi_slow_isi')
psi_fast_stc.save(savepath + subject + '/' + subject + '_psi_fast_isi')