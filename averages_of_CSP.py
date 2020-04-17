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
    sum_csp = mne.read_source_estimate(savepath + subject + '/' + subject + 'csp_sum_diff')

    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + subject + '/' + subject + 'morph_CSP-morph.h5')
    
    #Apply morph to SourceEstimate
    stc_fsaverage = morph.apply(sum_csp)
    
    n_vertices_fsave, n_times = stc_fsaverage.data.shape
    n_subjects = len(SUBJECTS)
    #    Let's make sure our results replicate, so set the seed.
    np.random.seed(0)
    X = np.random.randn(n_vertices_fsave, n_times, n_subjects) 
    X[:, :, :] += stc_fsaverage.data[:, :, np.newaxis]
    
X_avg = np.average(X,2)
#make stc format
X_CSP = stc_fsaverage
X_CSP.data = X_avg
#save
X_CSP.save(savepath + subject + '/'  + 'CSP_avg')