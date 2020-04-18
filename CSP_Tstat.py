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
from mne.stats import ttest_1samp_no_p

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
N=len(SUBJ_ASD)
ave=0.
ALL = []
for subject in SUBJ_ASD:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    
    #load stcs
    sum_csp_norm = mne.read_source_estimate(savepath + '1_results/CSP_sum/' + subject + 'sum_CSP_V3-V1_10_17Hz')

    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + '1_results/morph_CSP/' + subject + 'CSP-morph.h5')
    
    #Apply morph to SourceEstimate
    stc_fsaverage_csp_diff = morph.apply(sum_csp_norm)
    ALL.append(stc_fsaverage_csp_diff.data)
    ave += stc_fsaverage_csp_diff.data    
ave /= N
stc_fsaverage_csp_diff.data=ave
X = np.asarray(ALL)
Tval = ttest_1samp_no_p(X)
stc_Tval = stc_fsaverage_csp_diff
stc_Tval.data = Tval
# save stc Tstat
stc_Tval.save(savepath + '1_results/Tstat_CSP_sum/'  + '/' + 'ASD_Tstat_sum_CSP_diff_V3-V1') 

N=len(SUBJ_NT)
ave=0.
ALL = []
for subject in SUBJ_NT:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    
    #load stcs
    sum_csp_norm = mne.read_source_estimate(savepath + '1_results/CSP_sum/' + subject + 'sum_CSP_V3-V1_10_17Hz')

    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + '1_results/morph_CSP/' + subject + 'CSP-morph.h5')
    
    #Apply morph to SourceEstimate
    stc_fsaverage_csp_diff = morph.apply(sum_csp_norm)
    ALL.append(stc_fsaverage_csp_diff.data)
    ave += stc_fsaverage_csp_diff.data    
ave /= N
stc_fsaverage_csp_diff.data=ave
X = np.asarray(ALL)
Tval = ttest_1samp_no_p(X)
stc_Tval = stc_fsaverage_csp_diff
stc_Tval.data = Tval
# save stc Tstat
stc_Tval.save(savepath + '1_results/Tstat_CSP_sum/'  + '/' + 'NT_Tstat_sum_CSP_diff_V3-V1') 