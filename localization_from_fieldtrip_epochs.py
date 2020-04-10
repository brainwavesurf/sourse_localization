#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:58:55 2020

@author: a_shishkina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#get_ipython().run_line_magic('matplotlib', 'inline')
#matplotlib .use('TKAgg') 
#import packages
import mne
from mne import io
from mne.minimum_norm import make_inverse_operator, compute_source_psd_epochs

#load subj info
SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0379', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383']

SUBJECTS = SUBJ_ASD + SUBJ_NT
#SUBJECTS = ['0102']
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
    
    original_data = mne.io.read_raw_fif(raw_fname, preload=False)
    original_info = original_data.info
    
    #calculate noise covariance matrix from empty room data
    raw_fname = '/net/server/data/Archive/aut_gamma/orekhova/KI/EmptyRoom/' + subject + '/er/' + subject + '_er1_sss.fif'
    raw_noise = io.read_raw_fif(raw_fname, preload=True)
    raw_noise.filter(2, 40, fir_design='firwin') 
    methods = ['shrunk', 'empirical']
    noise_cov = mne.compute_raw_covariance(raw_noise, method=methods, rank=dict(meg=69)) 
    
    ftname = savepath + subject + '/' + subject + '_preproc_alpha_2_40_epochs.mat'
    #load data with all epochs from fieldtrip
    all_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='epochs', trialinfo_column=0)
    all_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_allepo.fif', overwrite=True)
    all_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_allepo.fif'
    all_epo = mne.read_epochs(all_fname, proj=False, verbose=None) 
    all_epo.crop(-0.8, 0)
    #inverse operator
    inverse_operator = make_inverse_operator(all_epo.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
        
    #load epochs for fast from fieldtrip
    fast_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='fast_alpha_bp', trialinfo_column=0)
    fast_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_allepo_fast.fif', overwrite=True)
    fast_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_allepo_fast.fif'
    fast_epo = mne.read_epochs(fast_fname, proj=False, verbose=None) 
    
    #load epochs for slow from fieldtrip
    slow_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='slow_alpha_bp', trialinfo_column=0)
    slow_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_allepo_slow.fif', overwrite=True)
    slow_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_allepo_slow.fif'
    slow_epo = mne.read_epochs(slow_fname, proj=False, verbose=None) 

    #calculate stcs
    method = "sLORETA"
    snr = 3.
    lambda2 = 1. / snr ** 2
    bandwidth = 'hann'

    #for fast 
    n_epochs_use = fast_epo.events.shape[0]
    stcs_fast = compute_source_psd_epochs(fast_epo[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_fast in enumerate(stcs_fast):
        psd_avg += stc_fast.data
    psd_avg /= n_epochs_use
    freqs = stc_fast.times  
    stc_fast.data = psd_avg  

    #for slow 
    n_epochs_use = slow_epo.events.shape[0]
    stcs_slow = compute_source_psd_epochs(slow_epo[:n_epochs_use], inverse_operator,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_slow in enumerate(stcs_slow):
        psd_avg += stc_slow.data
    psd_avg /= n_epochs_use
    freqs = stc_slow.times 
    stc_slow.data = psd_avg
    
    #save
    stc_slow.save(savepath + subject + '/' + subject + 'ft_slow_2_40Hz')
    stc_fast.save(savepath + subject + '/' + subject + 'ft_fast_2_40Hz')