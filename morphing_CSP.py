#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:40:31 2020

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

for subject in SUBJECTS:
    
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/'
    savepath = myPATH + 'Results_Alpha_and_Gamma/'
    
    #load stcs
    sum_csp = mne.read_source_estimate(savepath + '1_results/CSP_sum/' + subject + 'sum_CSP_V3-V1_10_17Hz')
    
    #Setting up SourceMorph for SourceEstimate
    morph = mne.compute_source_morph(sum_csp, subject_from='Case' + subject,
                                     subject_to='fsaverage5',
                                     subjects_dir=subjects_dir)
    
    #save
    morph.save(savepath + '1_results/morph_CSP/' + subject + 'CSP')