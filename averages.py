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
    
    n_vertices_fsave, n_times = stc_fsaverage_slow.data.shape
    n_subjects = len(SUBJECTS)
    #    Let's make sure our results replicate, so set the seed.
    np.random.seed(0)
    X = np.random.randn(n_vertices_fsave, n_times, n_subjects, 2) 
    X[:, :, :, 0] += stc_fsaverage_fast.data[:, :, np.newaxis]
    X[:, :, :, 1] += stc_fsaverage_slow.data[:, :, np.newaxis]
    
X = np.abs(X)  # only magnitude
X1 = (X[:, :, :, 0] - X[:, :, :, 1])/(X[:, :, :, 0] + X[:, :, :, 1])*100
X2 = X[:, :, :, 0] - X[:, :, :, 1]
X1_avg = np.average(X1,2)
X2_avg = np.average(X2,2)

X_with = stc_fsaverage_slow
X_with.data = X1_avg

X_without = stc_fsaverage_fast
X_without.data = X2_avg

X_with.save(savepath + subject + '/'  + 'avg_with_norm')
X_without.save(savepath + subject + '/' + 'avg_without_norm')