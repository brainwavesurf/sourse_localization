#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:50:14 2020

@author: mtw
"""

# Perform  permutation_cluster_test for V3 vs V1 in an averaged frequency range (tmin - tmax)
# Use paired ttest for unicariate comparisons
# Save/plot clusters (<0.05) and T-values over threshold


import numpy as np
from scipy import stats as stats
import mne
from mne.stats import  summarize_clusters_stc, permutation_cluster_test

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
subjects_dir = PATHfrom + 'freesurfersubjects/'
subjects_dir_case = PATHfrom + 'freesurfersubjects/Case'
savepath = myPATH + 'Results_Alpha_and_Gamma/'
savefolder = myPATH + 'Results_Alpha_and_Gamma/1_results/source_clusters/'
src_fname = subjects_dir + 'fsaverage5' + '/bem/fsaverage5-ico-5p-src.fif'
#read morphed stcs
V1 = []
V3 = []
for subject in SUBJECTS:  
    # Load stc to in common cortical space (fsaverage)
    #load stcs
    stc_slow = mne.read_source_estimate(savepath + subject + '/' + subject + 'meg_slow_isi_2_40Hz')
    stc_fast = mne.read_source_estimate(savepath + subject + '/' + subject + 'meg_fast_isi_2_40Hz')
    #Setting up SourceMorph for SourceEstimate
    morph = mne.read_source_morph(savepath + subject + '/' + subject + 'morph_2_40-morph.h5')
    #Apply morph to SourceEstimate
    stc_fsaverage_slow = morph.apply(stc_slow)
    stc_fsaverage_fast = morph.apply(stc_fast)
    
    V1.append(stc_fsaverage_slow.data[:,6:13])    
    V3.append(stc_fsaverage_fast.data[:,6:13])

V1 = np.mean(V1, axis=2)
V1 = V1[:, :, np.newaxis]     
V1 = np.transpose(V1, [0, 2, 1]) 
  
V3 = np.mean(V3, axis=2)
V3 = V3[:, :, np.newaxis]  
V3 = np.transpose(V3, [0, 2, 1])  
 
X = [V3, V1]
tstep = stc_fsaverage_slow.tstep 
# construct connectivity matrix
src = mne.read_source_spaces(src_fname)
fsave_vertices = [s['vertno'] for s in src]
connectivity = mne.spatial_src_connectivity(src)

#  Define stat function for univariate tests
def stat_fun(arg1, arg2):
    [statistic , p] = stats.ttest_rel(arg1, arg2) # no parallel computing # for related samples
    return (statistic);

# Do clustering
Nperm=1000
p_threshold = 0.05
df = V3[0].shape[1] + V1[0].shape[0] - 1
t_threshold = stats.distributions.t.ppf(1.-p_threshold/2, df)
T_obs, clusters, cluster_p_values, H0 = clu =\
     permutation_cluster_test (X, t_power=1, step_down_p=0.05, threshold=t_threshold, n_permutations=Nperm, stat_fun = stat_fun,  connectivity=connectivity, out_type='indices', n_jobs=1, tail=0)
    # If threshold is None, it will choose a t-threshold equivalent to  p < 0.05 for the given number of (within-subject) observations.
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

#  Plot/save T-stat 
T_obs = np.mean(T_obs, 0)
T_obs = T_obs.reshape(len(T_obs),1)
Tstc = stc_fsaverage_slow
Tstc.data = T_obs
Tstc.save(savefolder + 'ALL_Tstat_V3vsV1_ave' + str(10) + '_' + str(17) + 'Hz')


#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
if good_cluster_inds.shape[0]>0:
    stc_all_cluster_vis = summarize_clusters_stc(clu, p_thresh=0.05, #tstep=tstep,
                                                  vertices=fsave_vertices,
                                                  subject='fsaverage5')
    
    #save
    stc_all_cluster_vis.save(savefolder + 'ALL_Clusters0.05_V3>V1_red_' + str(10) + '_' + str(17) + 'Hz')