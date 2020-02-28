#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib 
#get_ipython().run_line_magic('matplotlib', 'inline')
#matplotlib .use('TKAgg') 
#import packages
import mne
from mne import io
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import numpy as np
from mne.report import Report
from mne.time_frequency import psd_array_welch
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, source_band_induced_power, compute_source_psd_epochs
import  matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
SUBJECTS = ['0106']
PATHfrom = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/'
subjects_dir = PATHfrom + 'freesurfersubjects'
subjects_dir_case = PATHfrom + 'freesurfersubjects/Case'
for subject in SUBJECTS:
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop'+ '/epochs/'
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
    
    ftname = savepath + subject + '/' + subject + '_fieldtrip_data_epochs.mat'
    
    #load csp data for CSP1 fast
    fast_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='epochs_fast1', trialinfo_column=0)
    fast_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + 'epo_fast1.fif')
    fast_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_csp_epo_fast1.fif'
    fast_epo = mne.read_epochs(fast_fname, proj=False, verbose=None) 
    
    #load csp data for CSP1 slow
    slow_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='epochs_slow1', trialinfo_column=0)
    slow_epo.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + 'epo_slow1.fif')
    slow_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + 'epo_slow1.fif'
    slow_epo = mne.read_epochs(slow_fname, proj=False, verbose=None) 
    
    #load bandpassed for fast poststim
    filename = subjpath + subject + '_preproc_alpha_bp_epochs.mat'
    epo_fast = mne.read_epochs_fieldtrip(filename, original_info, data_name='fast_alpha_post', trialinfo_column=0)
    epo_fast.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif')
    epo_fast_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif'
    fast_epo_post = mne.read_epochs(epo_fast_fname, proj=False, verbose=None) 
    
    #load bandpassed for fast poststim
    filename = subjpath + subject + '_preproc_alpha_bp_epochs.mat'
    epo_slow = mne.read_epochs_fieldtrip(filename, original_info, data_name='slow_alpha_post', trialinfo_column=0)
    epo_slow.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif')
    epo_slow_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif'
    slow_epo_post = mne.read_epochs(epo_slow_fname, proj=False, verbose=None)
    
    
    noise_cov_fast = mne.compute_covariance(fast_epo_post, method='shrinkage', rank=None)
    noise_cov_slow = mne.compute_covariance(slow_epo_post, method='shrinkage', rank=None)
    data_cov_fast = mne.compute_covariance(fast_epo, method='shrinkage', rank=None)
    data_cov_slow = mne.compute_covariance(slow_epo, method='shrinkage', rank=None)
    
    #fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov_fast, raw.info)
    #fig_cov, fig_spectra = mne.viz.plot_cov(data_cov_slow, raw.info)
    
    info = fast_epo.info
    inverse_operator_fast = make_inverse_operator(info, fwd, noise_cov_fast, loose=0.2, depth=0.8, verbose=True)
    
    info = slow_epo.info
    inverse_operator_slow = make_inverse_operator(info, fwd, noise_cov_slow, loose=0.2, depth=0.8, verbose=True)
    
    method = "MNE"
    snr = 3.
    lambda2 = 1. / snr ** 2
    bandwidth =  'hann'
    
    stc_fast = apply_inverse_epochs(fast_epo, inverse_operator_fast, lambda2, method=method,  pick_ori='normal')
    stc_slow = apply_inverse_epochs(slow_epo, inverse_operator_slow, lambda2, method=method,  pick_ori='normal')
    
    stc_fast_av=np.mean(stc_fast)
    stc_slow_av=np.mean(stc_slow)
    
    V_fast = []
    V_slow = []
    stc_fast = [stc_fast]
    stc_slow = [stc_slow]
    for s in stc_fast:
        temp = [element.data for element in s]
        V_fast.append(np.stack(temp))
    for s in stc_slow:    
        temp = [element.data for element in s]
        V_slow.append(np.stack(temp))
    
    #calculate spectral power on each epoch
    psds_fast, freqs = psd_array_welch(V_fast[0], sfreq=500, n_fft=512, n_overlap=0, n_per_seg = 400,fmin=10, fmax=17, n_jobs=1)
    psds_slow, freqs = psd_array_welch(V_slow[0], sfreq=500, n_fft=512, n_overlap=0, n_per_seg = 400,fmin=10, fmax=17, n_jobs=1)
    
    V_fast_av = psds_fast.mean(axis=0)
    V_slow_av = psds_slow.mean(axis=0)
    
    stc_my = mne.SourceEstimate(V_fast_av, [stc_fast_av.lh_vertno, stc_fast_av.rh_vertno], tmin=0, tstep=stc_fast_av.tstep, subject=subject)
    
    medium = np.max(V_fast_av[:, 1])/2
    maxim = np.max(V_fast_av[:, 1])
    
    savename_stc = savepath + subject + '/' + subject + '_stc_my'
    np.save(savename_stc, stc_my)
    savename_fast = savepath + subject + '/' + subject + '_V_fast_av'
    np.save(savename_fast, V_fast_av)
    savename_slow = savepath + subject + '/' + subject + '_V_slow_av'
    np.save(savename_slow, V_slow_av)
    
    np.load(savename_stc + '.npy', allow_pickle=True)
    np.load(savename_fast + '.npy', allow_pickle=True)
    np.load(savename_slow + '.npy', allow_pickle=True)
    
    brain_lat = stc_my.plot(subject='Case'+subject, initial_time=0.002, hemi='both', views='lat',  
                 clim=dict(kind='value', lims=(0, medium, maxim)), subjects_dir=subjects_dir, time_viewer = True)
    
    brain_cau = stc_my.plot(subject='Case'+subject, initial_time=0.002, hemi='both', views='cau',  
         clim=dict(kind='value', lims=(0, medium, maxim)), subjects_dir=subjects_dir)
    
    brain_med = stc_my.plot(subject='Case'+subject, initial_time=0.002, hemi='both', views='med',
         clim=dict(kind='value', lims=(0, medium, maxim)), subjects_dir=subjects_dir)
    
    brain_lat.save_image(savepath + subject + '/' + subject  + '_fast_lat.png')
    brain_cau.save_image(savepath + subject + '/' + subject  + '_fast_cau.png')
    brain_med.save_image(savepath + subject + '/' + subject  + '_fast_med.png')
    
    #add reports
    report = Report()
    report_fname = savepath + subject + '/' + subject  + '_fast_report.html'
    
    fig1 = mpimg.imread(savepath + subject + '/' + subject  + '_fast_lat.png')
    fig2 = mpimg.imread(savepath + subject + '/' + subject  + '_fast_cau.png')
    fig3 = mpimg.imread(savepath + subject + '/' + subject  + '_fast_med.png')
    
    report.add_figs_to_section(fig1, captions='Localization of fast activity', section= subject +' both LCMV_beamformer fast')  
    report.add_figs_to_section(fig2, captions='Localization of fast activity', section= subject +' both LCMV_beamformer fast')  
    report.add_figs_to_section(fig3, captions='Localization of fast activity', section= subject +' both LCMV_beamformer fast')  
    
     