#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:33:33 2020

@author: a_shishkina
"""

from itertools import product
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from spectral_connectivity import Multitaper, Connectivity

from spectral_connectivity.minimum_phase_decomposition import minimum_phase_decomposition
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

# all_subj_v1_v4_asd_slow = []
# all_subj_v4_v1_asd_slow = []
# all_subj_v1_v4_nt_slow = []
# all_subj_v4_v1_nt_slow = []

# all_subj_v1_v4_asd_medium = []
# all_subj_v4_v1_asd_medium = []
# all_subj_v1_v4_nt_medium = []
# all_subj_v4_v1_nt_medium = []

all_subj_v1_v4_asd_fast = []
all_subj_v4_v1_asd_fast = []
all_subj_v1_v4_nt_fast = []
all_subj_v4_v1_nt_fast = []

for subject in SUBJECTS:
 
    allepochs = mne.read_epochs(savepath + subject + '/' + subject + '_stim-epo.fif')
    # Sort epochs according to experimental conditions in the post-stimulus interval
    #slow_epo_stim = allepochs.__getitem__('V1')
    #medium_epo_stim = allepochs.__getitem__('V2') 
    fast_epo_stim = allepochs.__getitem__('V3')
    
    # epochs = [slow_epo_stim, medium_epo_stim, fast_epo_stim]
    # for epo in epochs:
    epo = fast_epo_stim
    
    fname_inv = savepath + subject + '/' + subject + '_inv'
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    src = inverse_operator['src'] 
    
    snr = 1.0  # use lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    
    # Read labels for V1 and v4 
    v1_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V1_rh.label')
    v4_label = mne.read_label(datapath + 'Results_Alpha_and_Gamma/' + subject + '/' + subject + '_V4_rh.label')
    
    stcs_v1 = apply_inverse_epochs(epo, inverse_operator, lambda2, method='sLORETA',
                                   pick_ori="normal", label=v1_label)
    stcs_v4 = apply_inverse_epochs(epo, inverse_operator, lambda2, method='sLORETA',
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
    
    sfreq = epo.info['sfreq'] 
    
    # Create signals input
    datarray = np.asarray(tcs_v1)
    signal_v1 = np.transpose(datarray.mean(2),(2,1,0)) #(times,epochs,signals))
    datarray = np.asarray(tcs_v4)
    signal_v4 = np.transpose(datarray.mean(2),(2,1,0)) 
    
    signal = np.append(signal_v1, signal_v4, axis=2)
    
    # Compute granger causality
    m = Multitaper(signal, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
    c = Connectivity(fourier_coefficients=m.fft()[:,:,:,0:130,:], frequencies=m.frequencies[0:130])
    
    cross_spectral_matrix = c._expectation(c._cross_spectral_matrix)
    n_signals = cross_spectral_matrix.shape[-1]
    total_power = c.power()
    n_frequencies = cross_spectral_matrix.shape[-3]
    non_neg_index = np.arange(0, (n_frequencies + 1) // 2)
    new_shape = list(cross_spectral_matrix.shape)
    new_shape[-3] = non_neg_index.size
    
    predictive_power = np.empty(new_shape)
    nvert_v1 = len(stc_label_v1[1].vertices)
    nvert_v4 = len(stc_label_v4[1].vertices)
    
    n_sign_v1 = np.arange(nvert_v1)
    n_sign_v4 = np.arange(nvert_v1, nvert_v1+nvert_v4)
    
    comb_signals = list(product(n_sign_v1,n_sign_v4))
    
    for pair_indices in comb_signals:
        pair_indices = np.array(pair_indices)[:, np.newaxis]
        try:
            minimum_phase_factor = (
                minimum_phase_decomposition(cross_spectral_matrix[..., pair_indices, pair_indices.T]))
            inverse_fourier_coefficients = ifft(minimum_phase_factor, axis=-3).real
            transfer_function = np.matmul(
                minimum_phase_factor, 
                np.linalg.inv(inverse_fourier_coefficients[..., 0:1, :, :]))[..., non_neg_index, :, :]
            noise_covariance = np.matmul(
                inverse_fourier_coefficients[..., 0, :, :],
                (inverse_fourier_coefficients[..., 0, :, :]).swapaxes(-1, -2).conjugate()).real
            variance = np.diagonal(noise_covariance, axis1=-1,
                                   axis2=-2)[..., np.newaxis]
            rotated_covariance = (variance.swapaxes(-1, -2) - noise_covariance ** 2 / variance)
           
            intrinsic_power = (total_power[..., pair_indices[:, 0]][..., np.newaxis] -
                               rotated_covariance[..., np.newaxis, :, :] *
                               np.abs(transfer_function) ** 2)
            
            intrinsic_power[intrinsic_power == 0] = np.finfo(float).eps
            pr_pow = total_power[..., pair_indices[:, 0]][..., np.newaxis] / intrinsic_power
            pr_pow[pr_pow <= 0] = np.nan
            
            predictive_power[..., pair_indices, pair_indices.T] = pr_pow
        except np.linalg.LinAlgError:
            predictive_power[
                ..., pair_indices, pair_indices.T] = np.nan
    
    diagonal_ind = np.diag_indices(n_signals)
    predictive_power[..., diagonal_ind[0], diagonal_ind[1]] = np.nan

    
    # Granger causality from v1 to v4
    m = np.empty((nvert_v1, nvert_v4, predictive_power.shape[1]))
    for i in np.arange(nvert_v1):
        for j in np.arange(nvert_v1, nvert_v1 + nvert_v4):
            m[i,j-nvert_v1,:] = predictive_power[...,i,j].squeeze()
    granger_avg_v1_v4 = m.mean(0).mean(0)
    
    
    # Granger causality from v4 to v1
    m = np.empty((nvert_v1, nvert_v4, predictive_power.shape[1]))
    for i in np.arange(nvert_v1):
        for j in np.arange(nvert_v1, nvert_v1 + nvert_v4):
            m[i,j-nvert_v1,:] = predictive_power[...,j,i].squeeze()
    granger_avg_v4_v1 = m.mean(0).mean(0)
    
    if subject in SUBJ_ASD:
        # all_subj_v1_v4_asd_slow.append(granger_avg_v1_v4)
        # all_subj_v4_v1_asd_slow.append(granger_avg_v4_v1)
  
        # all_subj_v1_v4_asd_medium.append(granger_avg_v1_v4)
        # all_subj_v4_v1_asd_medium.append(granger_avg_v4_v1)
    
        all_subj_v1_v4_asd_fast.append(granger_avg_v1_v4)
        all_subj_v4_v1_asd_fast.append(granger_avg_v4_v1)
    else:
   
        # all_subj_v1_v4_nt_slow.append(granger_avg_v1_v4)
        # all_subj_v4_v1_nt_slow.append(granger_avg_v4_v1)
    
        # all_subj_v1_v4_nt_medium.append(granger_avg_v1_v4)
        # all_subj_v4_v1_nt_medium.append(granger_avg_v4_v1)
   
        all_subj_v1_v4_nt_fast.append(granger_avg_v1_v4)
        all_subj_v4_v1_nt_fast.append(granger_avg_v4_v1)
            
    
# np.save(savepath + 'granger/' + 'all_v1_v4_nt_vert_slow', all_subj_v1_v4_nt_slow)
# np.save(savepath + 'granger/' + 'all_v4_v1_nt_vert_slow', all_subj_v4_v1_nt_slow)
# np.save(savepath + 'granger/' + 'all_v1_v4_asd_vert_slow', all_subj_v1_v4_asd_slow)
# np.save(savepath + 'granger/' + 'all_v4_v1_asd_vert_slow', all_subj_v4_v1_asd_slow)

# np.save(savepath + 'granger/' + 'all_v1_v4_nt_vert_medium', all_subj_v1_v4_nt_medium)
# np.save(savepath + 'granger/' + 'all_v4_v1_nt_vert_medium', all_subj_v4_v1_nt_medium)
# np.save(savepath + 'granger/' + 'all_v1_v4_asd_vert_medium', all_subj_v1_v4_asd_medium)
# np.save(savepath + 'granger/' + 'all_v4_v1_asd_vert_medium', all_subj_v4_v1_asd_medium)

np.save(savepath + 'granger/' + 'all_v1_v4_nt_vert_fast', all_subj_v1_v4_nt_fast)
np.save(savepath + 'granger/' + 'all_v4_v1_nt_vert_fast', all_subj_v4_v1_nt_fast)
np.save(savepath + 'granger/' + 'all_v1_v4_asd_vert_fast', all_subj_v1_v4_asd_fast)
np.save(savepath + 'granger/' + 'all_v4_v1_asd_vert_fast', all_subj_v4_v1_asd_fast)

stop = timeit.default_timer()
time = stop - start

# Load granger causality values for all subjects
all_subj_v1_v4_asd = np.load(savepath + 'granger/' + 'all_v1_v4_nt_vert_slow.npy')
all_subj_v1_v4_nt = np.load(savepath + 'granger/' + 'all_v1_v4_nt_vert_slow.npy')
all_subj_v4_v1_asd = np.load(savepath + 'granger/' + 'all_v4_v1_asd_vert_slow.npy')
all_subj_v4_v1_nt = np.load(savepath + 'granger/' + 'all_v4_v1_nt_vert_slow.npy')

# Average over subjects
avg_subj_v1_v4_asd = all_subj_v1_v4_asd.mean(0)
avg_subj_v1_v4_nt = all_subj_v1_v4_nt.mean(0)
avg_subj_v4_v1_asd = all_subj_v4_v1_asd.mean(0)
avg_subj_v4_v1_nt = all_subj_v4_v1_nt.mean(0)

# Plot
plt.plot(c.frequencies, avg_subj_v1_v4_asd[0,:], label='v1->v4')
plt.plot(c.frequencies, avg_subj_v4_v1_asd[0,:], label='v4->v1')
plt.legend()
plt.ylim(0,0.05)
plt.xlim(2,40)
plt.title('asd')
plt.ylabel('Granger Causaliy Value')
plt.xlabel('frequency')
plt.savefig(savepath + 'granger/' + 'all_v1_v4_asd_vert_slow')

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
