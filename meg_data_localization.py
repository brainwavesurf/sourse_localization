#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:23:01 2020

@author: a_shishkina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import matplotlib 
#get_ipython().run_line_magic('matplotlib', 'inline')
#matplotlib .use('TKAgg') 
#import packages
import mne
from mne import io
from mne.minimum_norm import make_inverse_operator, compute_source_psd_epochs
import scipy.io
#load subj info
SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0379', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383'] 

SUBJECTS = SUBJ_ASD + SUBJ_NT
SUBJECTS = ['0106']
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
        
    #raw epochs case
    fif_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '-noerror-lagcorrected-epo.fif'
    
    #load info about preceding events
    info_mat = scipy.io.loadmat(savepath + subject + '/' + subject + '_info.mat')
    info_file = info_mat['allinfo']['prev_stim_type'][0][0][0]
    
    #interstimulus epochs
    epo_isi = mne.read_epochs(fif_fname, proj=False, verbose=None) 
    epo_isi.filter(2,40)
    epo_isi.events[:,2] = info_file
    epo_isi.crop(tmin=-0.8, tmax=0)
    slow_epo_isi = epo_isi.__getitem__('V1')
    medium_epo_isi = epo_isi.__getitem__('V2')
    fast_epo_isi = epo_isi.__getitem__('V3')

    #stimulation epochs
    epo_post = mne.read_epochs(fif_fname, proj=False, verbose=None) 
    epo_post.filter(2,40)
    epo_post.events[:,2] = info_file
    epo_post.crop(tmin=0.4, tmax=1.2)
    slow_epo_post = epo_post.__getitem__('V1')
    medium_epo_post = epo_post.__getitem__('V2')
    fast_epo_post = epo_post.__getitem__('V3')
    
    #calculate noise covariance matrix from empty room data
    raw_fname = '/net/server/data/Archive/aut_gamma/orekhova/KI/EmptyRoom/' + subject + '/er/' + subject + '_er1_sss.fif'
    raw_noise = io.read_raw_fif(raw_fname, preload=True)
    raw_noise.filter(2, 40, fir_design='firwin') 
    methods = ['shrunk', 'empirical']
    noise_cov = mne.compute_raw_covariance(raw_noise, method=methods, rank=dict(meg=69)) 
    
    #slow
    inverse_operator_slow_isi = make_inverse_operator(slow_epo_isi.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
    inverse_operator_slow_post = make_inverse_operator(slow_epo_post.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)  
    #medium
    inverse_operator_medium_isi = make_inverse_operator(medium_epo_isi.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
    inverse_operator_medium_post = make_inverse_operator(medium_epo_post.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)      
    #fast
    inverse_operator_fast_isi = make_inverse_operator(fast_epo_isi.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
    inverse_operator_fast_post = make_inverse_operator(fast_epo_post.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)
    
    method = "sLORETA"
    snr = 3.
    #lambda2 = 0.05
    lambda2 = 1. / snr ** 2
    bandwidth =  4.0
    
    #for slow interstimulus epochs
    n_epochs_use = slow_epo_isi.events.shape[0]
    stcs_slow_isi = compute_source_psd_epochs(slow_epo_isi[:n_epochs_use], inverse_operator_slow_isi,
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

    #for slow stimulation period epochs
    n_epochs_use = slow_epo_post.events.shape[0]
    stcs_slow_post = compute_source_psd_epochs(slow_epo_post[:n_epochs_use], inverse_operator_slow_post,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_slow_post in enumerate(stcs_slow_post):
        psd_avg += stc_slow_post.data
    psd_avg /= n_epochs_use
    freqs = stc_slow_post.times  # the frequencies are stored here
    stc_slow_post.data = psd_avg 

    #for medium interstimulus epochs
    n_epochs_use = medium_epo_isi.events.shape[0]
    stcs_medium_isi = compute_source_psd_epochs(medium_epo_isi[:n_epochs_use], inverse_operator_medium_isi,
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

    #for medium stimulation period epochs
    n_epochs_use = medium_epo_post.events.shape[0]
    stcs_medium_post = compute_source_psd_epochs(medium_epo_post[:n_epochs_use], inverse_operator_medium_post,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_medium_post in enumerate(stcs_medium_post):
        psd_avg += stc_medium_post.data
    psd_avg /= n_epochs_use
    freqs = stc_medium_post.times
    stc_medium_post.data = psd_avg 
    
    #for fast interstimulus epochs
    n_epochs_use = fast_epo_isi.events.shape[0]
    stcs_fast_isi = compute_source_psd_epochs(fast_epo_isi[:n_epochs_use], inverse_operator_fast_isi,
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

    #for fast stimulation period epochs
    n_epochs_use = fast_epo_post.events.shape[0]
    stcs_fast_post = compute_source_psd_epochs(fast_epo_post[:n_epochs_use], inverse_operator_fast_post,
                                 lambda2=lambda2,
                                 method=method, fmin=2, fmax=40,
                                 bandwidth=bandwidth,
                                 return_generator=True, verbose=True)
    psd_avg = 0.
    for i, stc_fast_post in enumerate(stcs_fast_post):
        psd_avg += stc_fast_post.data
    psd_avg /= n_epochs_use
    freqs = stc_fast_post.times  
    stc_fast_post.data = psd_avg 

    #subtract "baseline" (stimulation period) from interstimulus data  
    stc_slow = stc_slow_isi
    stc_slow.data = (stc_slow_isi.data - stc_slow_post.data)/stc_slow_post.data  
    
    stc_medium = stc_medium_isi
    stc_medium.data = (stc_medium_isi.data - stc_medium_post.data)/stc_medium_post.data  
    
    stc_fast = stc_fast_isi
    stc_fast.data = (stc_fast_isi.data - stc_fast_post.data)/stc_fast_post.data  
      
    #save
    stc_slow.save(savepath + subject + '/' + subject + 'meg_slow')
    stc_medium.save(savepath + subject + '/' + subject + 'meg_medium')
    stc_fast.save(savepath + subject + '/' + subject + 'meg_fast')