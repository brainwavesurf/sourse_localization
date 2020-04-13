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
           '0378', '0379', '0384']
                       
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
    slow_isi = mne.read_source_estimate(savepath + subject + '/' + subject + 'meg_slow_isi_2_40Hz')
    medium_isi = mne.read_source_estimate(savepath + subject + '/' + subject + 'meg_medium_isi_2_40Hz')
    fast_isi = mne.read_source_estimate(savepath + subject + '/' + subject + 'meg_fast_isi_2_40Hz')
    
    #Setting up SourceMorph for SourceEstimate
    morph_slow = mne.compute_source_morph(slow_isi, subject_from='Case' + subject,
                                     subject_to='fsaverage',
                                     subjects_dir=subjects_dir)
    
    morph_medium = mne.compute_source_morph(medium_isi, subject_from='Case' + subject,
                                     subject_to='fsaverage',
                                     subjects_dir=subjects_dir)
    
    morph_fast = mne.compute_source_morph(fast_isi, subject_from='Case' + subject,
                                     subject_to='fsaverage',
                                     subjects_dir=subjects_dir)
    #Apply morph to SourceEstimate
    stc_fsaverage_slow = morph_slow.apply(slow_isi)
    stc_fsaverage_medium = morph_medium.apply(medium_isi)
    stc_fsaverage_fast = morph_fast.apply(fast_isi)
    
    #save
    morph_slow.save(savepath + subject + '/' + subject + 'morph_slow_2_40')
    morph_medium.save(savepath + subject + '/' + subject + 'morph_medium_2_40')
    morph_fast.save(savepath + subject + '/' + subject + 'morph_fast_2_40')