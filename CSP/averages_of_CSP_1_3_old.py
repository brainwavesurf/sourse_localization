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

#for the NT group
X_sum = []
for subject in SUBJ_NT:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    diff = []
    CSP = ['1','2','3']
    for num in CSP:
        stc_diff = mne.read_source_estimate(savepath + subject + '/' + subject + 'csp_V3-V1_old' + num)      
        diff.append(stc_diff.data)        
    #sum and save
    diff_sum_1_3 = stc_diff
    diff_sum_1_3.data = sum(diff)


    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + '1_results/morph_CSP/' + subject + 'CSP-morph.h5')
    
    #Apply morph to SourceEstimate
    sum_csp_fsaverage = morph.apply(diff_sum_1_3)
    
    X_sum.append(sum_csp_fsaverage.data)
    
#average across freqs and subjects
X = np.asarray(X_sum)
X_avg_freq = np.mean(X, axis=2)
X_avg_group = np.mean(X_avg_freq, axis=0)
X_avg = X_avg_group[:, np.newaxis]

#make stc format and save
avg_CSP_sum_NT = sum_csp_fsaverage
avg_CSP_sum_NT.data = X_avg
avg_CSP_sum_NT.save(savepath + '1_results/average_CSP_sum/' + 'NT_avg_sum_CSP_1_3_old')

#for the ASD group
X_sum = []
for subject in SUBJ_ASD:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    diff = []
    CSP = ['1','2','3']
    for num in CSP:
        stc_diff = mne.read_source_estimate(savepath + subject + '/' + subject + 'csp_V3-V1_old' + num)      
        diff.append(stc_diff.data)        
    #sum and save
    diff_sum_1_3 = stc_diff
    diff_sum_1_3.data = sum(diff)

    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + '1_results/morph_CSP/' + subject + 'CSP-morph.h5')
    
    #Apply morph to SourceEstimate
    sum_csp_fsaverage = morph.apply(diff_sum_1_3)
    
    X_sum.append(sum_csp_fsaverage.data)
    
#average across freqs and subjects
X_avg_freq = np.mean(X_sum, axis=2)
X_avg_group = np.mean(X_avg_freq, axis=0)
X_avg = X_avg_group[:, np.newaxis]

#make stc format and save
avg_CSP_sum_ASD = sum_csp_fsaverage
avg_CSP_sum_ASD.data = X_avg
avg_CSP_sum_ASD.save(savepath + '1_results/average_CSP_sum/' + 'ASD_avg_sum_CSP_1_3_old')

avg_CSP_sum_group_diff = sum_csp_fsaverage
avg_CSP_sum_group_diff.data = avg_CSP_sum_NT.data - avg_CSP_sum_ASD.data
avg_CSP_sum_group_diff.save(savepath + '1_results/average_CSP_sum/' + 'NTvsASD_avg_sum_CSP_1_3_old')