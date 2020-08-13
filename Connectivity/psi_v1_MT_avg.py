#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#phase slope index computation between 20 vertices with max power in V1 and 5 vertices with max power in MT

"""
Created on Tue Jul 22 14:37:57 2020

@author: a_shishkina
"""
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator, source_band_induced_power
from mne.connectivity import phase_slope_index
from mne import io

import matplotlib.pyplot as plt

import numpy as np
import scipy.io

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
    
    # Resample epochs
    if allepochs.info['sfreq']>500:
        allepochs.resample(500)
        
    # Sort epochs according to experimental conditions in the post-stimulus interval
    allepochs.crop(tmin=-0.8, tmax=0)
    slow_epo_isi = allepochs.__getitem__('V1')
    fast_epo_isi = allepochs.__getitem__('V3')
    
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
    
    # Compute a source estimate for 10-17 Hz frequency band
    stcs_slow_v1_induced = source_band_induced_power(slow_epo_isi, inverse_operator, dict(alpha_beta=[10,17]), method='sLORETA', n_cycles=2, 
                                                     n_jobs=1, use_fft=True, label=v1_label)
    
    stcs_slow_mt_induced = source_band_induced_power(fast_epo_isi, inverse_operator, dict(alpha_beta=[10,17]), method='sLORETA', n_cycles=2, 
                                                     n_jobs=1, use_fft=True, label=mt_label)
    
    
    # plt.plot(stcs_slow_v1_induced['alpha_beta'].vertices[1], stcs_slow_v1_induced['alpha_beta'].data.mean(1), label='slow')
    # plt.plot(stcs_fast_v1_induced['alpha_beta'].vertices[1], stcs_fast_v1_induced['alpha_beta'].data.mean(1), label='fast')
    # plt.legend()
    
    #sort array of vertices in descending order according power
    pow_idx_v1 = stcs_slow_v1_induced['alpha_beta'].data.mean(1).argsort()
    vert_descend_v1 = stcs_slow_v1_induced['alpha_beta'].vertices[1][pow_idx_v1[::-1]]
    #take only 20 vertices with max power
    v1_label.vertices=vert_descend_v1[0:20]
    
    #same for mt
    pow_idx_mt = stcs_slow_mt_induced['alpha_beta'].data.mean(1).argsort()
    vert_descend_mt = stcs_slow_mt_induced['alpha_beta'].vertices[1][pow_idx_mt[::-1]]
    #take only 20 vertices with max power
    mt_label.vertices=vert_descend_mt[0:5]
    
    #extract time courses from vertices
    src = inverse_operator['src']  # the source space used
    seed_ts_slow_v1 = mne.extract_label_time_course(stcs_slow_v1, v1_label, src, mode='mean_flip', verbose='error')
    seed_ts_slow_mt = mne.extract_label_time_course(stcs_slow_mt, mt_label, src, mode='mean_flip', verbose='error')
    
    comb_ts_slow = list(zip(seed_ts_slow_v1, seed_ts_slow_mt))

    indices = (np.array([0]), np.array([1]))
    
    # Compute the PSI in the frequency range 11Hz-17Hz.
    fmin = 10.
    fmax = 17.
    sfreq = slow_epo_isi.info['sfreq']  # the sampling frequency
    
    psi_slow, freqs, times, n_epochs, _ = phase_slope_index(
        comb_ts_slow, mode='fourier', sfreq=sfreq, indices=indices,
        fmin=fmin, fmax=fmax)
    avg_v1_mt.append(psi_slow)
np.save(savepath + 'psi/' + 'all_v1_mt_max_pow_rh', avg_v1_mt)

#load PSIs for avg time series and max time series
all_avg = np.load(savepath + 'psi/' + 'all_v1_mt_avg_flip_rh.npy')
all_max = np.load(savepath + 'psi/' + 'all_v1_mt_max_pow_rh.npy')

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.scatter(np.arange(42), all_avg.mean(2).mean(1))
ax1.set_title('Averaged time courses')
ax1.plot(np.arange(42), np.zeros(42), 'r--')
ax1.set_ylim(-0.3,0.3)
ax1.set_ylabel('PSI')
ax2.scatter(np.arange(42), all_max.mean(2).mean(1), c='green')
ax2.set_title('Max power time courses')
ax2.plot(np.arange(42), np.zeros(42), 'r--')
ax2.set_ylim(-0.3,0.3)
ax2.set_ylabel('PSI')
ax2.set_xlabel('Subjects')
plt.savefig(savepath + 'psi/' + 'avgVSmax')