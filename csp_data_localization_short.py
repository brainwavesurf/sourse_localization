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
#0163-trans, many subj bem-innerskull
#SUBJECTS = SUBJ_ASD + SUBJ_NT
SUBJECTS = ['0159','0140','0358']
PATHfrom = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/'
subjects_dir = PATHfrom + 'freesurfersubjects'
subjects_dir_case = PATHfrom + 'freesurfersubjects/Case'
for subject in SUBJECTS:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = PATHfrom + 'Results_Alpha_and_Gamma/'
    
    #create bem model and make its solution
    conductivity = (0.3,)  # for single layer
    model = mne.make_bem_model(subject='Case'+subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    del model
    raw_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw.fif'
    raw = io.Raw(raw_fname, preload=True)
    del raw
    #set up a source space (inflated brain); if volume - use pos=5
    src = mne.setup_source_space('Case' + subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
    trans = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw-trans.fif'
    
    #make forward solution
    fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=2)
    del bem, src
    #download 2-40 Hz filtered data from fieldtrip
    #ftname = savepath + subject + '/' + subject + '_preproc_alpha_2_40_epochs.mat'
    
    #raw epochs case
    fif_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '-noerror-lagcorrected-epo.fif'
    #interstimulus epochs
    epo_isi = mne.read_epochs(fif_fname, proj=False, verbose=None) 
    epo_isi.crop(tmin=-0.8, tmax=0)
    #stimulation epochs
    epo_post = mne.read_epochs(fif_fname, proj=False, verbose=None) 
    epo_post.crop(tmin=0.4, tmax=1.2)
    
    #load csp data for CSP1 fast from fieldtrip
    #ftname = savepath + subject + '/' + subject + '_fieldtrip_data_epochs.mat'
    #fast_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='fast_alpha_bp', trialinfo_column=0)
    #fast_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif')
    #fast_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif'
    #fast_epo = mne.read_epochs(fast_fname, proj=False, verbose=None) 
    
    #load csp data for CSP1 slow from fieldtrip
    #slow_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='slow_alpha_bp', trialinfo_column=0)
    #slow_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif')
    #slow_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif'
    #slow_epo = mne.read_epochs(slow_fname, proj=False, verbose=None) 
    
    #calculate noise covariance matrix from empty room data
    raw_fname = '/net/server/data/Archive/aut_gamma/orekhova/KI/EmptyRoom/' + subject + '/er/' + subject + '_er1_sss.fif'
    raw_noise = io.read_raw_fif(raw_fname, preload=True)
    raw_noise.filter(2, 40, fir_design='firwin') 
    noise_cov = mne.compute_raw_covariance(raw_noise, method='shrinkage')
    
    #calculare inverse operator
    #inverse_operator_fast = make_inverse_operator(fast_epo.info, fwd, noise_cov, loose=0.2, depth=None, verbose=True)
    #inverse_operator_slow = make_inverse_operator(slow_epo.info, fwd, noise_cov, loose=0.2, depth=None, verbose=True)
    
    inverse_operator_isi = make_inverse_operator(epo_isi.info, fwd, noise_cov, loose=0.2, depth=None, verbose=True)
    inverse_operator_post = make_inverse_operator(epo_post.info, fwd, noise_cov, loose=0.2, depth=None, verbose=True)
    
    
    method = "sLORETA"
    #snr = 3.
    lambda2 = 0.05
    #1. / snr ** 2
    bandwidth =  4.0
    
    #for slow epochs
#    n_epochs_use = 49
#    stcs_slow = compute_source_psd_epochs(slow_epo[:n_epochs_use], inverse_operator_slow,
#                                 lambda2=lambda2,
#                                 method=method, fmin=10, fmax=17,
#                                 bandwidth=bandwidth,
#                                 return_generator=True, verbose=True)
#
#    psd_avg = 0.
#    for i, stc_slow in enumerate(stcs_slow):
#        psd_avg += stc_slow.data
#    psd_avg /= n_epochs_use
#    freqs = stc_slow.times  # the frequencies are stored here
#    stc_slow.data = psd_avg  
    
    #for fast epochs
#    n_epochs_use = 38
#    stcs_fast = compute_source_psd_epochs(fast_epo[:n_epochs_use], inverse_operator_fast,
#                                 lambda2=lambda2,
#                                 method=method, fmin=10, fmax=17,
#                                 bandwidth=bandwidth,
#                                 return_generator=True, verbose=True)
#  
#    psd_avg = 0.
#    for i, stc_fast in enumerate(stcs_fast):
#        psd_avg += stc_fast.data
#    psd_avg /= n_epochs_use
#    freqs = stc_fast.times  # the frequencies are stored here
#    stc_fast.data = psd_avg  
#    
    
    #for interstimulus epochs
    n_epochs_use = epo_isi.events.shape[0]
    stcs_isi = compute_source_psd_epochs(epo_isi[:n_epochs_use], inverse_operator_isi,
                                 lambda2=lambda2,
                                 method=method, fmin=10, fmax=17,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_isi in enumerate(stcs_isi):
        psd_avg += stc_isi.data
    psd_avg /= n_epochs_use
    freqs = stc_isi.times  # the frequencies are stored here
    stc_isi.data = psd_avg  

    #for stimulation period epochs
    n_epochs_use = epo_post.events.shape[0]
    stcs_post = compute_source_psd_epochs(epo_post[:n_epochs_use], inverse_operator_post,
                                 lambda2=lambda2,
                                 method=method, fmin=10, fmax=17,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_post in enumerate(stcs_post):
        psd_avg += stc_post.data
    psd_avg /= n_epochs_use
    freqs = stc_post.times  # the frequencies are stored here
    stc_post.data = psd_avg

    #subtract "baseline" (stimulation period) from interstimulus data 
    stc = stc_isi
    stc.data = abs(stc_isi.data - stc_post.data)/stc_post.data
    
    medium = (np.max(stc.data[:,3]))/2
    maxim = np.max(stc.data[:,3])
    brain = stc.plot(subject='Case'+subject, initial_time=14.9626, hemi='both', views='caud',  # 10 HZ
                 clim=dict(kind='value', lims=(0, medium, maxim)), subjects_dir=subjects_dir)
    #time_viewer = True
    brain.save_image(savepath + subject + '/' + subject + '_15HZ.png')
    del brain
    