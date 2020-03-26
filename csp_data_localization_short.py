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

#load subj info
SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0379', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383'] 

#File "/net/server/data/Archive/aut_gamma/orekhova/KI/freesurfersubjects/Case0107/bem/inner_skull.surf" does not exist

SUBJECTS = SUBJ_ASD + SUBJ_NT
PATHfrom = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
myPATH = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/'
subjects_dir = PATHfrom + 'freesurfersubjects'
subjects_dir_case = PATHfrom + 'freesurfersubjects/Case'
for subject in SUBJECTS:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    
    trans = PATHfrom + 'TRANS/' + subject + '_rings_ICA_raw-trans.fif'
    
    #create bem model and make its solution
    conductivity = (0.3,)  # for single layer
    model = mne.make_bem_model(subject='Case'+subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    del model
    raw_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw.fif'
   
    #set up a source space (inflated brain); if volume - use pos=5
    src = mne.setup_source_space('Case' + subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
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
    epo_isi_slow = epo_isi["V1"]
    epo_isi_fast = epo_isi["V3"]
    
    #stimulation epochs
    epo_post = mne.read_epochs(fif_fname, proj=False, verbose=None) 
    epo_post.crop(tmin=0.4, tmax=1.2)
    epo_post_slow = epo_post["V1"]
    epo_post_fast = epo_post["V3"]
    

    #calculate noise covariance matrix from empty room data
    raw_fname = '/net/server/data/Archive/aut_gamma/orekhova/KI/EmptyRoom/' + subject + '/er/' + subject + '_er1_sss.fif'
    raw_noise = io.read_raw_fif(raw_fname, preload=True)
    raw_noise.filter(2, 40, fir_design='firwin') 
    noise_cov = mne.compute_raw_covariance(raw_noise, method='shrinkage')
    
    #slow
    inverse_operator_isi_slow = make_inverse_operator(epo_isi_slow.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
    inverse_operator_post_slow = make_inverse_operator(epo_post_slow.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
   
    #fast
    inverse_operator_isi_fast = make_inverse_operator(epo_isi_fast.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
    inverse_operator_post_fast = make_inverse_operator(epo_post_fast.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
      
    method = "sLORETA"
    snr = 3.
    #lambda2 = 0.05
    lambda2 = 1. / snr ** 2
    bandwidth =  4.0

    #for fast interstimulus epochs
    n_epochs_use = epo_isi_fast.events.shape[0]
    stcs_isi_fast = compute_source_psd_epochs(epo_isi_fast[:n_epochs_use], inverse_operator_isi_fast,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_isi_fast in enumerate(stcs_isi_fast):
        psd_avg += stc_isi_fast.data
    psd_avg /= n_epochs_use
    freqs = stc_isi_fast.times  # the frequencies are stored here
    stc_isi_fast.data = psd_avg  

    #for fast stimulation period epochs
    n_epochs_use = epo_post_fast.events.shape[0]
    stcs_post_fast = compute_source_psd_epochs(epo_post_fast[:n_epochs_use], inverse_operator_post_fast,
                                 lambda2=lambda2,
                                 method=method, fmin=10, fmax=17,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_post_fast in enumerate(stcs_post_fast):
        psd_avg += stc_post_fast.data
    psd_avg /= n_epochs_use
    freqs = stc_post_fast.times  # the frequencies are stored here
    stc_post_fast.data = psd_avg
#    
    #for slow interstimulus epochs
    n_epochs_use = epo_isi_slow.events.shape[0]
    stcs_isi_slow = compute_source_psd_epochs(epo_isi_slow[:n_epochs_use], inverse_operator_isi_slow,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_isi_slow in enumerate(stcs_isi_slow):
        psd_avg += stc_isi_slow.data
    psd_avg /= n_epochs_use
    freqs = stc_isi_slow.times  # the frequencies are stored here
    stc_isi_slow.data = psd_avg  

    #for slow stimulation period epochs
    n_epochs_use = epo_post_slow.events.shape[0]
    stcs_post_slow = compute_source_psd_epochs(epo_post_slow[:n_epochs_use], inverse_operator_post_slow,
                                 lambda2=lambda2,
                                 method=method, fmin=10, fmax=17,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_post_slow in enumerate(stcs_post_slow):
        psd_avg += stc_post_slow.data
    psd_avg /= n_epochs_use
    freqs = stc_post_slow.times  # the frequencies are stored here
    stc_post_slow.data = psd_avg

    #subtract "baseline" (stimulation period) from interstimulus data 
    stc_fast = stc_isi_fast
    stc_fast.data = (stc_isi_fast.data - stc_post_fast.data)/stc_post_slow.data  
    
    stc_slow = stc_isi_slow
    stc_slow.data = (stc_isi_slow.data - stc_post_slow)/stc_post_fast.data
    
    #plot
    medium = (np.max(stc_fast.data[:,3]))/2
    maxim = np.max(stc_fast.data[:,3])
    brain_fast = stc_fast.plot(subject='Case'+subject, initial_time=14.9626, hemi='both', views='caud',  # 10 HZ
                 clim=dict(kind='value', lims=(0, medium, maxim)), subjects_dir=subjects_dir)
    brain_fast.save_image(savepath + subject + '/' + subject + 'isi_post_fast.png')
    del brain_fast
    
    medium = (np.max(stc_slow.data[:,3]))/2
    maxim = np.max(stc_slow.data[:,3])
    brain_slow = stc_slow.plot(subject='Case'+subject, initial_time=14.9626, hemi='both', views='caud',  # 10 HZ
                 clim=dict(kind='value', lims=(0, medium, maxim)), subjects_dir=subjects_dir)
    
    #time_viewer = True
    brain_slow.save_image(savepath + subject + '/' + subject + 'isi_post_slow.png')
    del brain_slow
    