#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:58:55 2020

@author: a_shishkina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
#import packages
import mne
from mne.minimum_norm import make_inverse_operator, compute_source_psd_epochs

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
    #load noise covariance matrix from empty room data
    noise_cov = mne.read_cov(savepath + subject + '/' + subject + 'noise_cov_10_17Hz', verbose=None)
    
    original_data = mne.io.read_raw_fif(raw_fname, preload=False)
    original_info = original_data.info
    
    #for the first 3 CSP components and the second 3 CSP components (commented)
    diff = []
    CSP = ['1','2','3']
    #CSP = ['4','5','6']
    for num in CSP:
        
        #load csp data for fast from fieldtrip
        ftname = savepath + subject + '/' + subject + '_fieldtrip_csp_1_6.mat'
        fast_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='epochs_fast'+num, trialinfo_column=0)
        fast_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif', overwrite=True)
        fast_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif'
        fast_epo = mne.read_epochs(fast_fname, proj=False, verbose=None) 
        
        #load csp data for slow from fieldtrip
        slow_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='epochs_slow'+num, trialinfo_column=0)
        slow_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif', overwrite=True)
        slow_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif'
        slow_epo = mne.read_epochs(slow_fname, proj=False, verbose=None) 

        #inverse operator
        inverse_operator_fast = make_inverse_operator(fast_epo.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
        inverse_operator_slow = make_inverse_operator(slow_epo.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
        
        method = "sLORETA"
        snr = 3.
        lambda2 = 1. / snr ** 2
        bandwidth = 'hann'
    
        #for fast csp
        n_epochs_use = fast_epo.events.shape[0]
        stcs_fast = compute_source_psd_epochs(fast_epo[:n_epochs_use], inverse_operator_fast,
                                     lambda2=lambda2,
                                     method=method, fmin=10, fmax=17,
                                     bandwidth=bandwidth,
                                     return_generator=True, verbose=True)
        psd_avg = 0.
        for i, stc_fast in enumerate(stcs_fast):
            psd_avg += stc_fast.data
        psd_avg /= n_epochs_use
        freqs = stc_fast.times  # the frequencies are stored here
        stc_fast.data = psd_avg  

        #for slow csp
        n_epochs_use = slow_epo.events.shape[0]
        stcs_slow = compute_source_psd_epochs(slow_epo[:n_epochs_use], inverse_operator_slow,
                                     lambda2=lambda2,
                                     method=method, fmin=10, fmax=17,
                                     bandwidth=bandwidth,
                                     return_generator=True, verbose=True)
        psd_avg = 0.
        for i, stc_slow in enumerate(stcs_slow):
            psd_avg += stc_slow.data
        psd_avg /= n_epochs_use
        freqs = stc_slow.times  # the frequencies are stored here
        stc_slow.data = psd_avg
        
        #subtract slow from fast power 
        stc_diff = stc_fast
        stc_diff_norm = stc_slow
        
        stc_diff.data = stc_fast.data - stc_slow.data
        #save
        stc_diff.save(savepath + subject + '/' + subject + 'csp_V3-V1' + num)      
        diff.append(stc_diff.data)
        
    #sum
    diff_sum = stc_slow
    diff_sum.data = sum(diff)
    
    #save
    diff_sum.save(savepath + '1_results/CSP_sum/' + subject + 'sum_CSP_1_3_V3-V1')
    #diff_sum.save(savepath + '1_results/CSP_sum/' + subject + 'sum_CSP_4_6_V3-V1')
    