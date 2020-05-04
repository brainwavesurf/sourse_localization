#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:19:31 2020

@author: a_shishkina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import packages
import mne
import numpy as np
from scipy import stats as stats
from mne.stats import summarize_clusters_stc, permutation_cluster_test


#load subj info
SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383'] 

PATHfrom = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
myPATH = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/'
subjects_dir = PATHfrom + 'freesurfersubjects/'
src_fname = subjects_dir + 'fsaverage5' + '/bem/fsaverage5-ico-5p-src.fif'
savefolder = myPATH + 'Results_Alpha_and_Gamma/1_results/source_clusters/'
# 'fsaverage5/bem/fsaverage-ico-5-src.fif'
tmin=11.
tmax=13.

X1=[]
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

    stcV1 = stc_fsaverage_slow.crop(tmin=tmin, tmax=tmax, include_tmax=True)   
    stcV3 = stc_fsaverage_fast.crop(tmin=tmin, tmax=tmax, include_tmax=True)
    
    stcDiff = (stcV3-stcV1)/(stcV3+stcV1)
    stcDiff=stcDiff.mean()
    X1.append(stcDiff.data)

X1  =  np.array(X1)  
X1 = np.transpose(X1, [0, 2, 1])

X2=[]
for subj in SUBJ_ASD:
    
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

    stcV1 = stc_fsaverage_slow.crop(tmin=tmin, tmax=tmax, include_tmax=True)   
    stcV3 = stc_fsaverage_fast.crop(tmin=tmin, tmax=tmax, include_tmax=True)
    
    stcDiff = (stcV3-stcV1)/(stcV3+stcV1)
    stcDiff=stcDiff.mean()
    X2.append(stcDiff.data)
    
tstep = stcV1.tstep     
X2  =  np.array(X2)  
X2 = np.transpose(X2, [0, 2, 1])
X = [X1, X2]

src = mne.read_source_spaces(src_fname)
#fsave_vertices = [src[0]['vertno'], src[1]['vertno']] 
fsave_vertices = [s['vertno'] for s in src]
#fsave_vertices = [src[0]['vertno']] 
connectivity = mne.spatial_src_connectivity(src)

# Define statistical function for univariate analysis
def stat_fun(arg1, arg2):
    #[statistic , p] = stats.mannwhitneyu(arg1, arg2) #cannot slice a 0-d array  # comparing two related samples
    #wilcoxon(arg1, arg2) # for unrelated samples
    [statistic , p] = stats.ttest_ind(arg1, arg2) # no parallel computing # for related samples
    return (statistic);

#run clustering
Nperm=1000

# Define T-threshol 
p_threshold = 0.05
df =  X1.shape[0] + X2.shape[0] - 2
t_threshold = stats.distributions.t.ppf(1.-p_threshold/2, df)

T_obs, clusters, cluster_p_values, H0 = clu =\
     permutation_cluster_test (X, threshold=t_threshold, n_permutations=Nperm, stat_fun = stat_fun,  connectivity=connectivity, out_type='indices', n_jobs=1, tail=0)
    # If threshold is None, it will choose a t-threshold equivalent to  p < 0.05 for the given number of (within-subject) observations.
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]


# Plot T statistics
T_obs = np.transpose(T_obs, [1,0])
Tstc=stcV1
Tstc.data= T_obs
######################## save stc
Tstc.save(savefolder + 'Tstat_NT>ASDmean_11_17Hz_cluster')   


# ... and plot clusters if any
#good_cluster_inds = np.where(cluster_p_values < 0.5)[0]
# We do not have 'good' clusters, take  non-significant to test the plot:
if good_cluster_inds.shape[0]>0:
    stc_all_cluster_vis = summarize_clusters_stc(clu, p_thresh=0.05, tstep=tstep,
                                                  vertices=fsave_vertices,
                                                  subject='fsaverage')
    

    stc_all_cluster_vis.save(savefolder + 'NTvsASD_mean_11_17Hz_cluster')