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
X_sum = []
X_sum_norm = []
X_avg = []
X_avg_norm = []
for subject in SUBJ_NT:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    
    #load stcs
    sum_csp = mne.read_source_estimate(savepath + '1_results/CSP_sum/' + subject + 'sum_CSP_V3-V1_10_17Hz')

    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + subject + '/' + subject + 'morph_CSP-morph.h5')
    
    #Apply morph to SourceEstimate
    sum_csp_fsaverage = morph.apply(sum_csp)
    #save
    #sum_csp_fsaverage.save(savepath + '1_results/CSP_sum_fsaverage5/' + subject + 'sum_CSP_V3-V1_10_17Hz')
    
    X_sum.append(sum_csp_fsaverage.data)
    
#average across subjects   
X_avg_sum = sum(X_sum)/len(X_sum)

#make stc format and save
avg_CSP_sum = sum_csp_fsaverage
avg_CSP_sum.data = X_avg_sum
avg_CSP_sum.save(savepath + '1_results/average_CSP_sum/' + 'NT_avg_sum_CSP')

X_sum = []
X_sum_norm = []
X_avg = []
X_avg_norm = []
for subject in SUBJ_ASD:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    
    #load stcs
    sum_csp = mne.read_source_estimate(savepath + '1_results/CSP_sum/' + subject + 'sum_CSP_V3-V1_10_17Hz')

    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + subject + '/' + subject + 'morph_CSP-morph.h5')
    
    #Apply morph to SourceEstimate
    sum_csp_fsaverage = morph.apply(sum_csp)
    #save
    #sum_csp_fsaverage.save(savepath + '1_results/CSP_sum_fsaverage5/' + subject + 'sum_CSP_V3-V1_10_17Hz')
    
    X_sum.append(sum_csp_fsaverage.data)
    
#average across subjects   
X_avg_sum = sum(X_sum)/len(X_sum)

#make stc format and save
avg_CSP_sum = sum_csp_fsaverage
avg_CSP_sum.data = X_avg_sum
avg_CSP_sum.save(savepath + '1_results/average_CSP_sum/' + 'ASD_avg_sum_CSP')