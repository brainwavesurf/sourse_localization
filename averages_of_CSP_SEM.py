#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:05:14 2020

@author: a_shishkina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import packages
import mne
from scipy import stats
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

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
    
#average over subjects
avg = np.mean(X_sum, axis=0)
np.save(savepath + '1_results/average_CSP_sum/' + 'NT_avgoversubj_CSP', avg)

#calculate the standart error of the mean 
stand = np.std(X_sum, axis=0)
hand_sem = stand/sqrt(len(SUBJ_ASD))
np.save(savepath + '1_results/average_CSP_sum/' + 'NT_handSEM_CSP', hand_sem)
   
X_sum_SEM = stats.sem(X_sum, axis=0)
np.save(savepath + '1_results/average_CSP_sum/' + 'NT_SEM_CSP', X_sum_SEM)

#plot histograms
for i in np.arange(1,6):
    plt.figure(i)
    plt.subplot(1, 2, 1)
    plt.hist(X_sum_SEM[:,i-1])
    plt.xticks([min(X_sum_SEM[:,i-1]), np.median(X_sum_SEM[:,i-1]), max(X_sum_SEM[:,i-1])])

    plt.subplot(1, 2, 2)
    plt.hist(avg[:,i-1],color = "skyblue")
    plt.xticks([min(avg[:,i-1]), np.median(avg[:,i-1]), max(avg[:,i-1])])
    plt.show()

#make stc format and save
sem_CSP_sum = sum_csp_fsaverage
sem_CSP_sum.data = X_sum_SEM
sem_CSP_sum.save(savepath + '1_results/average_CSP_sum/' + 'NT_SEM_sum_CSP')

X_sum = []
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
  
#average over subjects
avg = np.mean(X_sum, axis=0)
np.save(savepath + '1_results/average_CSP_sum/' + 'ASD_avgoversubj_CSP', avg)

#calculate the standart error of the mean 
stand = np.std(X_sum, axis=0)
hand_sem = stand/sqrt(len(SUBJ_ASD))
np.save(savepath + '1_results/average_CSP_sum/' + 'ASD_handSEM_CSP', hand_sem)
   
X_sum_SEM = stats.sem(X_sum, axis=0)
np.save(savepath + '1_results/average_CSP_sum/' + 'ASD_SEM_CSP', X_sum_SEM)

#make stc format and save
sem_CSP_sum = sum_csp_fsaverage
sem_CSP_sum.data = X_sum_SEM
sem_CSP_sum.save(savepath + '1_results/average_CSP_sum/' + 'ASD_SEM_sum_CSP')