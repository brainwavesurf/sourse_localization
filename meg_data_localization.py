#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:19:17 2020

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
    #download 2-40 Hz filtered data from fieldtrip
    #ftname = savepath + subject + '/' + subject + '_preproc_alpha_2_40_epochs.mat'
    
#    #raw epochs case
#    fif_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '-noerror-lagcorrected-epo.fif'
#    #interstimulus epochs
#    epo_isi = mne.read_epochs(fif_fname, proj=False, verbose=None) 
#    epo_isi.crop(tmin=-0.8, tmax=0)
#    epo_isi_slow = epo_isi["V1"]
#    epo_isi_fast = epo_isi["V3"]
#    
#    #stimulation epochs
#    epo_post = mne.read_epochs(fif_fname, proj=False, verbose=None) 
#    epo_post.crop(tmin=0.4, tmax=1.2)
#    epo_post_slow = epo_post["V1"]
#    epo_post_fast = epo_post["V3"]
    
    original_data = mne.io.read_raw_fif(raw_fname, preload=False)
    original_info = original_data.info
    
    #load slow epochs from fieldtrip
    ftname = savepath + subject + '/' + subject + '_preproc_epochs_2_40.mat'
    slow_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='slow_epochs', trialinfo_column=0)
    slow_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + 'slow_epo.fif', overwrite=True)
    slow_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + 'slow_epo.fif'
    #slow condition in interstimulus
    slow_epo_isi = mne.read_epochs(slow_fname, proj=False, verbose=None) 
    slow_epo_isi.crop(tmin=-0.8, tmax=0)
    #slow condition in stimulation
    slow_epo_post = mne.read_epochs(slow_fname, proj=False, verbose=None) 
    slow_epo_post.crop(tmin=0.4, tmax=1.2)
    
    #load medium epochs from fieldtrip
    ftname = savepath + subject + '/' + subject + '_preproc_epochs_2_40.mat'
    medium_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='medium_epochs', trialinfo_column=0)
    medium_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + 'medium_epo.fif', overwrite=True)
    medium_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + 'medium_epo.fif'
    #medium condition in interstimulus
    medium_epo_isi = mne.read_epochs(medium_fname, proj=False, verbose=None) 
    medium_epo_isi.crop(tmin=-0.8, tmax=0)
    #medium condition in stimulation
    medium_epo_post = mne.read_epochs(medium_fname, proj=False, verbose=None) 
    medium_epo_post.crop(tmin=0.4, tmax=1.2)
    
    #load fast epochs from fieldtrip
    ftname = savepath + subject + '/' + subject + '_preproc_epochs_2_40.mat'
    fast_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='fast_epochs', trialinfo_column=0)
    fast_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + 'fast_epo.fif', overwrite=True)
    fast_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + 'fast_epo.fif'
    #fast condition in interstimulus
    fast_epo_isi = mne.read_epochs(fast_fname, proj=False, verbose=None) 
    fast_epo_isi.crop(tmin=-0.8, tmax=0)
    #fast condition in stimulation
    fast_epo_post = mne.read_epochs(fast_fname, proj=False, verbose=None) 
    fast_epo_post.crop(tmin=0.4, tmax=1.2)
    


    #calculate noise covariance matrix from empty room data
    raw_fname = '/net/server/data/Archive/aut_gamma/orekhova/KI/EmptyRoom/' + subject + '/er/' + subject + '_er1_sss.fif'
    raw_noise = io.read_raw_fif(raw_fname, preload=True)
    raw_noise.filter(2, 40, fir_design='firwin') 
    noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical', rank=None)
    #methods = ['shrunk', 'empirical']
    #noise_cov = mne.compute_raw_covariance(raw_noise, method=methods, return_estimators=True)
    #rank = mne.compute_rank(noise_cov[0], info = raw_noise.info)
    #fig1, fig2 = noise_cov[0].plot(raw_noise.info)
    
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
    freqs = stc_slow_isi.times  # the frequencies are stored here
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
    freqs = stc_medium_isi.times  # the frequencies are stored here
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
    freqs = stc_medium_post.times  # the frequencies are stored here
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
    freqs = stc_fast_isi.times  # the frequencies are stored here
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
    freqs = stc_fast_post.times  # the frequencies are stored here
    stc_fast_post.data = psd_avg 


    #subtract "baseline" (stimulation period) from interstimulus data  
    stc_slow = stc_slow_isi
    stc_slow.data = abs(stc_slow_isi.data - stc_slow_post.data)/stc_slow_post.data  
    
    stc_medium = stc_medium_isi
    stc_medium.data = abs(stc_medium_isi.data - stc_medium_post.data)/stc_medium_post.data  
    
    stc_fast = stc_fast_isi
    stc_fast.data = abs(stc_fast_isi.data - stc_fast_post.data)/stc_fast_post.data  
    
    #plot
#    medium = (np.max(stc_fast.data[:,3]))/2
#    maxim = np.max(stc_fast.data[:,3])
#    brain_fast = stc_fast.plot(subject='Case'+subject, initial_time=14.9626, hemi='both', views='caud',  # 10 HZ
#                 clim=dict(kind='value', lims=(0, medium, maxim)), subjects_dir=subjects_dir)
#    brain_fast.save_image(savepath + subject + '/' + subject + 'isi_post_fast.png')
#    del brain_fast
#    
#    medium = (np.max(stc_slow.data[:,3]))/2
#    maxim = np.max(stc_slow.data[:,3])
#    brain_slow = stc_slow.plot(subject='Case'+subject, initial_time=14.9626, hemi='both', views='caud',  # 10 HZ
#                 clim=dict(kind='value', lims=(0, medium, maxim)), subjects_dir=subjects_dir) 
#    time_viewer = True
#    brain_slow.save_image(savepath + subject + '/' + subject + 'isi_post_slow.png')
#    del brain_slow
    
    #save
    stc_slow.save(savepath + subject + '/' + subject + 'meg_slow')
    stc_medium.save(savepath + subject + '/' + subject + 'meg_medium')
    stc_fast.save(savepath + subject + '/' + subject + 'meg_fast')