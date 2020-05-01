#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:05:14 2020

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
#for all subjects
X = []
for subject in SUBJECTS:
    
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
    
    stc_str = morph.apply(stc_slow)
    
    stc_fsaverage_diff = (stc_fsaverage_fast.data[:,6:13] - stc_fsaverage_slow.data[:,6:13])/(stc_fsaverage_fast.data[:,6:13] + stc_fsaverage_slow.data[:,6:13])*100    
    
    X.append(stc_fsaverage_diff)
    

X_avg_freq = np.mean(X, axis=2)
X_avg_group = np.mean(X_avg_freq, axis=0)
X_avg = X_avg_group[:, np.newaxis]

X_abs = stc_str
X_abs.data = X_avg
X_abs.save(savepath + '1_results/average_meg_diff/' + '10_17_avg_%_diff')

#ASD group
X = []
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
    
    stc_str = morph.apply(stc_slow)
    
    stc_fsaverage_diff = (stc_fsaverage_fast.data[:,6:13] - stc_fsaverage_slow.data[:,6:13])/(stc_fsaverage_fast.data[:,6:13] + stc_fsaverage_slow.data[:,6:13])*100    
    
    X.append(stc_fsaverage_diff)
    
X_avg_freq = np.mean(X, axis=2)
X_avg_group = np.mean(X_avg_freq, axis=0)
X_avg = X_avg_group[:, np.newaxis]

X_abs_ASD = stc_str
X_abs_ASD.data = X_avg
X_abs_ASD.save(savepath + '1_results/average_meg_diff/' + 'ASD_10_17_avg_%_diff')

#NT group
X = []
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
    
    stc_str = morph.apply(stc_slow)
    
    stc_fsaverage_diff = (stc_fsaverage_fast.data[:,6:13] - stc_fsaverage_slow.data[:,6:13])/(stc_fsaverage_fast.data[:,6:13] + stc_fsaverage_slow.data[:,6:13])*100
    
    X.append(stc_fsaverage_diff)
    
X_avg_freq = np.mean(X, axis=2)
X_avg_group = np.mean(X_avg_freq, axis=0)
X_avg = X_avg_group[:, np.newaxis]

X_abs_NT = stc_str
X_abs_NT.data = X_avg
X_abs_NT.save(savepath + '1_results/average_meg_diff/' + 'NT_10_17_avg_%_diff')

X_diff = stc_fsaverage_slow
X_diff.data = X_abs_NT.data - X_abs_ASD.data
X_diff.save(savepath + '1_results/average_meg_diff/' + 'NTvsASD_10_17_avg_%_diff')