#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:19:17 2020

@author: a_shishkina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib 
#get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib .use('TKAgg') 
#import packages
import mne
from mne import io
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import numpy as np
from mne.time_frequency import psd_array_welch
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, source_band_induced_power, compute_source_psd_epochs
import  matplotlib.pyplot as plt


##load subj info
#SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
#           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
#           '0378', '0379', '0384']
#                       
#SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
#            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
#            '0276', '0346', '0347', '0351', '0358', 
#            '0380', '0381', '0382', '0383'] 
#SUBJECTS = SUBJ_ASD + SUBJ_NT
SUBJECTS = ['0141']
PATHfrom = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/'
subjects_dir = PATHfrom + 'freesurfersubjects'
subjects_dir_case = PATHfrom + 'freesurfersubjects/Case'
for subject in SUBJECTS:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = PATHfrom + 'Results_Alpha_and_Gamma/'
    
    epoch_type = '-lagcorrected-epo.fif'
    #create bem model and make its solution
    conductivity = (0.3,)  # for single layer
    model = mne.make_bem_model(subject='Case'+subject, ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    raw_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw.fif'
    
    raw = io.Raw(raw_fname, preload=True)
    picks = mne.pick_types(raw.info,  meg='planar1')
    #set up a source space (inflated brain); if volume - use pos=5
    src = mne.setup_source_space('Case'+subject, spacing='oct6',
                                subjects_dir=subjects_dir, add_dist=False)
    
    trans = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw-trans.fif'
    
    #make forward solution
    fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.0, n_jobs=2)
    
    original_data = mne.io.read_raw_fif(raw_fname, preload=False)
    original_info = original_data.info
    
    ftname = savepath + subject + '/' + subject + '_preproc_alpha_2_40_epochs.mat'
    fif_fname = '/net/server/data/Archive/aut_gamma/orekhova/KI/SUBJECTS/0141/ICA_nonotch_crop/epochs/0141-noerror-lagcorrected-epo.fif'
    fast_epo_fif = mne.read_epochs(fif_fname, proj=False, verbose=None) 
    epo_fif= mne.Epochs(fif_fname, tmin=-0.8, tmax=0)

    #load csp data for CSP1 fast
    fast_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='fast_alpha_bp', trialinfo_column=0)
    fast_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif')
    fast_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif'
    fast_epo = mne.read_epochs(fast_fname, proj=False, verbose=None) 
    
    #load csp data for CSP1 slow
    slow_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='slow_alpha_bp', trialinfo_column=0)
    slow_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif')
    slow_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif'
    slow_epo = mne.read_epochs(slow_fname, proj=False, verbose=None) 
    
    raw_fname = '/net/server/data/Archive/aut_gamma/orekhova/KI/EmptyRoom/' + subject + '/er/' + subject + '_er1_sss.fif'
    raw_noise = io.read_raw_fif(raw_fname, preload=True)
    raw_noise.filter(2, 40, fir_design='firwin') 
    noise_cov = mne.compute_raw_covariance(raw_noise, method='shrinkage')
    
#    savename_inv_fast = savepath + subject + '/' + subject + '_inv_fast'
#    savename_inv_slow = savepath + subject + '/' + subject + '_inv_slow'
    
    info = fast_epo.info
    inverse_operator_fast = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
    
    info = slow_epo.info
    inverse_operator_slow = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
    
    info = fast_epo_fif.info
    inverse_operator_fif = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)

    method = "eLORETA"
    snr = 3.
    lambda2 = 1. / snr ** 2
    bandwidth =  'hann'
    n_epochs_use = 49
    stcs_slow = compute_source_psd_epochs(slow_epo[:n_epochs_use], inverse_operator_slow,
                                 lambda2=lambda2,
                                 method=method, fmin=8, fmax=13,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)

    psd_avg = 0.
    for i, stc_slow in enumerate(stcs_slow):
        psd_avg += stc_slow.data
    psd_avg /= n_epochs_use
    freqs = stc_slow.times  # the frequencies are stored here
    stc_slow.data = psd_avg  
    
    n_epochs_use = 38
    stcs_fast = compute_source_psd_epochs(fast_epo[:n_epochs_use], inverse_operator_fast,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
  
    psd_avg = 0.
    for i, stc_fast in enumerate(stcs_fast):
        psd_avg += stc_fast.data
    psd_avg /= n_epochs_use
    freqs = stc_fast.times  # the frequencies are stored here
    stc_fast.data = psd_avg  
    
    n_epochs_use = 184
    stcs_fif = compute_source_psd_epochs(fast_epo_fif[:n_epochs_use], inverse_operator_fif,
                                 lambda2=lambda2,
                                 method=method, fmin=8, fmax=13,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)

    psd_avg = 0.
    for i, stc_fif in enumerate(stcs_fif):
        psd_avg += stc_fif.data
    psd_avg /= n_epochs_use
    freqs = stc_fif.times  # the frequencies are stored here
    stc_fif.data = psd_avg  


    stc = stc_fast
    stc.data = stc_slow.data - stc_fast.data
    
    medium = (np.max(stc_fif.data[:,2]))/2
    maxim = np.max(stc_fif.data[:,2])
    brain = stc_fif.plot(subject='Case'+subject, initial_time=11.2219, hemi='both', views='lat',  # 10 HZ
                 clim=dict(kind='value', lims=(0, medium, maxim)), subjects_dir=subjects_dir, time_viewer = True)
    
    brain.save_image(savepath + subject + '/' + subject + '_fast.png')
    brain.save_single_image(savepath + subject + '/' + subject + '_fast.png')
    