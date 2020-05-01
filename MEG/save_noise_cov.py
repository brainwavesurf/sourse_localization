#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:38:38 2020

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

SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0379', '0384']
                       
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
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    #calculate noise covariance matrix from empty room data filtered 10-17
    raw_fname = '/net/server/data/Archive/aut_gamma/orekhova/KI/EmptyRoom/' + subject + '/er/' + subject + '_er2_sss.fif'
    raw_noise = io.read_raw_fif(raw_fname, preload=True)
    raw_noise.filter(10, 17, fir_design='firwin') 
    raw_noise.crop(0, 80)
    methods = ['shrunk', 'empirical']
    noise_cov = mne.compute_raw_covariance(raw_noise, method=methods, rank=dict(meg=69))
    noise_cov.save(savepath + subject + '/' + subject + 'noise_cov_10_17Hz')
    
    #calculate noise covariance matrix from empty room data filtered 2-40
    raw_fname = '/net/server/data/Archive/aut_gamma/orekhova/KI/EmptyRoom/' + subject + '/er/' + subject + '_er2_sss.fif'
    raw_noise = io.read_raw_fif(raw_fname, preload=True)
    raw_noise.filter(2, 40, fir_design='firwin') 
    raw_noise.crop(0, 80)
    methods = ['shrunk', 'empirical']
    noise_cov = mne.compute_raw_covariance(raw_noise, method=methods, rank=dict(meg=69))
    noise_cov.save(savepath + subject + '/' + subject + 'noise_cov_2_40Hz')