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
from mne import io
from mne.minimum_norm import make_inverse_operator, compute_source_psd_epochs
import scipy.io
import numpy as np

#load subj info
SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383'] 

SUBJECTS = SUBJ_ASD + SUBJ_NT

PATHfrom = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
myPATH = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/'
subjects_dir = PATHfrom + 'freesurfersubjects'
subjects_dir_case = PATHfrom + 'freesurfersubjects/Case'
for subject in SUBJECTS:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    
    raw_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw.fif'
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
    info_mat = scipy.io.loadmat(PATHfrom + 'Results_Alpha_and_Gamma/'+ subject + '/' + subject + '_info.mat')
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
    medium_epo_isi = allepochs.__getitem__('V2')
    fast_epo_isi = allepochs.__getitem__('V3')
    
    #load noise covariance matrix from empty room data
    noise_cov = mne.read_cov(savepath + subject + '/' + subject + 'noise_cov_2_40Hz', verbose=None)
    
    #inverse operator
    inverse_operator = make_inverse_operator(allepochs.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)

    #stc calculation    
    method = "sLORETA"
    snr = 3.
    #lambda2 = 0.05
    lambda2 = 1. / snr ** 2
    bandwidth = 'hann'
    
    #for slow interstimulus epochs
    n_epochs_use = slow_epo_isi.events.shape[0]
    stcs_slow_isi = compute_source_psd_epochs(slow_epo_isi[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_slow_isi in enumerate(stcs_slow_isi):
        psd_avg += stc_slow_isi.data
    psd_avg /= n_epochs_use
    freqs = stc_slow_isi.times  
    stc_slow_isi.data = psd_avg  

    #for medium interstimulus epochs
    n_epochs_use = medium_epo_isi.events.shape[0]
    stcs_medium_isi = compute_source_psd_epochs(medium_epo_isi[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_medium_isi in enumerate(stcs_medium_isi):
        psd_avg += stc_medium_isi.data
    psd_avg /= n_epochs_use
    freqs = stc_medium_isi.times  
    stc_medium_isi.data = psd_avg
    
    #for fast interstimulus epochs
    n_epochs_use = fast_epo_isi.events.shape[0]
    stcs_fast_isi = compute_source_psd_epochs(fast_epo_isi[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_fast_isi in enumerate(stcs_fast_isi):
        psd_avg += stc_fast_isi.data
    psd_avg /= n_epochs_use
    freqs = stc_fast_isi.times  
    stc_fast_isi.data = psd_avg

    #save
    stc_slow_isi.save(savepath + subject + '/' + subject + 'meg_slow_isi_2_40Hz')
    stc_medium_isi.save(savepath + subject + '/' + subject + 'meg_medium_isi_2_40Hz')
    stc_fast_isi.save(savepath + subject + '/' + subject + 'meg_fast_isi_2_40Hz')