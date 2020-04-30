#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:23:01 2020

@author: a_shishkina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import packages
import mne
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

#ASD group
X_slow = []
X_fast = []
for subject in SUBJ_ASD:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    
    #load stcs
    stc_slow = mne.read_source_estimate(savepath + subject + '/' + subject + 'meg_slow_isi_2_40Hz')
    stc_fast = mne.read_source_estimate(savepath + subject + '/' + subject + 'meg_fast_isi_2_40Hz')

    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + subject + '/' + subject + 'morph_2_40-morph.h5')
    
    #Apply morph to SourceEstimate
    stc_fsaverage_slow = morph.apply(stc_slow)
    stc_fsaverage_fast = morph.apply(stc_fast)
       
    X_slow.append(stc_fsaverage_slow.data[:,6:13])
    X_fast.append(stc_fsaverage_fast.data[:,6:13])
    
X_slow_avg_freq = np.mean(X_slow, axis=2)
X_slow_avg_group = np.mean(X_slow_avg_freq, axis=0)
X_slow_avg = X_slow_avg_group[:, np.newaxis]

ASD_X_slow = stc_fsaverage_slow
ASD_X_slow.data = X_slow_avg
ASD_X_slow.save(savepath + '1_results/average_meg_conditions/' + 'V1_ASD_10_17_avg')

X_fast_avg_freq = np.mean(X_fast, axis=2)
X_fast_avg_group = np.mean(X_fast_avg_freq, axis=0)
X_fast_avg = X_fast_avg_group[:, np.newaxis]

ASD_X_fast = stc_fsaverage_fast
ASD_X_fast.data = X_fast_avg
ASD_X_fast.save(savepath + '1_results/average_meg_conditions/' + 'V3_ASD_10_17_avg')

#NT group
X_slow = []
X_fast = []
for subject in SUBJ_NT:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    
    #load stcs
    stc_slow = mne.read_source_estimate(savepath + subject + '/' + subject + 'meg_slow_isi_2_40Hz')
    stc_fast = mne.read_source_estimate(savepath + subject + '/' + subject + 'meg_fast_isi_2_40Hz')

    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + subject + '/' + subject + 'morph_2_40-morph.h5')
    
    #Apply morph to SourceEstimate
    stc_fsaverage_slow = morph.apply(stc_slow)
    stc_fsaverage_fast = morph.apply(stc_fast)
       
    X_slow.append(stc_fsaverage_slow.data[:,6:13])
    X_fast.append(stc_fsaverage_fast.data[:,6:13])
    
X_slow_avg_freq = np.mean(X_slow, axis=2)
X_slow_avg_group = np.mean(X_slow_avg_freq, axis=0)
X_slow_avg = X_slow_avg_group[:, np.newaxis]

#
NT_X_slow = stc_fsaverage_slow
NT_X_slow.data = X_slow_avg
NT_X_slow.save(savepath + '1_results/average_meg_conditions/' + 'V1_NT_10_17_avg')

X_fast_avg_freq = np.mean(X_fast, axis=2)
X_fast_avg_group = np.mean(X_fast_avg_freq, axis=0)
X_fast_avg = X_fast_avg_group[:, np.newaxis]

NT_X_fast = stc_fsaverage_fast
NT_X_fast.data = X_fast_avg
NT_X_fast.save(savepath + '1_results/average_meg_conditions/' + 'V3_NT_10_17_avg')

#difference between groups
DIFF_X_slow = stc_fsaverage_slow
DIFF_X_slow.data = NT_X_slow.data - ASD_X_slow.data
DIFF_X_slow.save(savepath + '1_results/average_meg_conditions/' + 'V1_NT-ASD_10_17_avg')

DIFF_X_fast = stc_fsaverage_fast
DIFF_X_fast.data = NT_X_fast.data - ASD_X_fast.data
DIFF_X_fast.save(savepath + '1_results/average_meg_conditions/' + 'V3_NT-ASD_10_17_avg')