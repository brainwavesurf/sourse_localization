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
#for all subjects
X_with = []
X_without = []
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
    stc_str1 = morph.apply(stc_slow)
    stc_str2 = morph.apply(stc_fast)
    n_vertices_fsave, n_times = stc_fsaverage_slow.data.shape
    
    stc_fsaverage_diff_with = ((stc_fsaverage_fast.data - stc_fsaverage_slow.data)/(stc_fsaverage_fast.data + stc_fsaverage_slow.data))*100
    stc_fsaverage_diff_without = stc_fsaverage_fast.data - stc_fsaverage_slow.data
    
    X_with.append(stc_fsaverage_diff_with)
    X_without.append(stc_fsaverage_diff_without)
    
X_with_avg = sum(X_with)/len(X_with)
X_without_avg = sum(X_without)/len(X_without)

X_with = stc_str1
X_with.data = X_with_avg

X_without = stc_str2
X_without.data = X_without_avg

X_with.save(savepath + '1_results/average_meg_diff/' + '/' + '2_40avg_with_norm')
X_without.save(savepath + '1_results/average_meg_diff/' + '/' + '2_40avg_without_norm')

#ASD group
X_with = []
X_without = []
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
    
    stc_str1 = morph.apply(stc_slow)
    stc_str2 = morph.apply(stc_fast)
    
    stc_fsaverage_diff_with = ((stc_fsaverage_fast.data - stc_fsaverage_slow.data)/(stc_fsaverage_fast.data + stc_fsaverage_slow.data))*100
    stc_fsaverage_diff_without = stc_fsaverage_fast.data - stc_fsaverage_slow.data
    
    X_with.append(stc_fsaverage_diff_with)
    X_without.append(stc_fsaverage_diff_without)
    
X_with_avg = sum(X_with)/len(X_with)
X_without_avg = sum(X_without)/len(X_without)

X_with = stc_str1
X_with.data = X_with_avg

X_without = stc_str2
X_without.data = X_without_avg

X_with.save(savepath + '1_results/average_meg_diff/' + 'ASD_10_17_avg_with_norm')
X_without.save(savepath + '1_results/average_meg_diff/' +  'ASD_2_40avg_without_norm')

#NT group
X_with = []
X_without = []
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
    stc_str1 = morph.apply(stc_slow)
    stc_str2 = morph.apply(stc_fast)
    
    stc_fsaverage_diff_with = ((stc_fsaverage_fast.data - stc_fsaverage_slow.data)/(stc_fsaverage_fast.data + stc_fsaverage_slow.data))*100
    stc_fsaverage_diff_without = stc_fsaverage_fast.data - stc_fsaverage_slow.data
    
    X_with.append(stc_fsaverage_diff_with)
    X_without.append(stc_fsaverage_diff_without)
    
X_with_avg = sum(X_with)/len(X_with)
X_without_avg = sum(X_without)/len(X_without)

X_with = stc_str1
X_with.data = X_with_avg

X_without = stc_str2
X_without.data = X_without_avg

X_with.save(savepath + '1_results/average_meg_diff/' + 'NT_2_40avg_with_norm')
X_without.save(savepath + '1_results/average_meg_diff/' + 'NT_2_40avg_without_norm')