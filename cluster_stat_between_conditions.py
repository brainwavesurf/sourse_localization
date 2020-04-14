#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:16 2020

@author: a_shishkina
"""

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
from mne import spatial_tris_connectivity, grade_to_tris
import numpy as np
from scipy import stats as stats
from mne.stats import summarize_clusters_stc, spatio_temporal_cluster_test


#load subj info
SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383'] 

SUBJECTS = SUBJ_NT + SUBJ_ASD
PATHfrom = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
myPATH = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/'
subjects_dir = PATHfrom + 'freesurfersubjects'

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
    
    n_vertices_fsave, n_times = stc_fsaverage_slow.data.shape
    n_subjects = len(SUBJECTS)
    #    Let's make sure our results replicate, so set the seed.
    np.random.seed(0)
    X1 = np.random.randn(n_vertices_fsave, n_times, n_subjects)
    X1[:, :, :] += stc_fsaverage_slow.data[:, :, np.newaxis]
    X2 = np.random.randn(n_vertices_fsave, n_times, n_subjects)
    X2[:, :, :] += stc_fsaverage_fast.data[:, :, np.newaxis]
    
#We want to compare the overall activity levels for each condition
X1 = np.abs(X1)  # only magnitude
X2 = np.abs(X2)  # only magnitude
    
X1 = np.transpose(X1, [2, 1, 0])
X2 = np.transpose(X2, [2, 1, 0])

connectivity = spatial_tris_connectivity(grade_to_tris(5))

p_threshold = 0.0001
f_threshold = stats.distributions.f.ppf(1. - p_threshold / 2.,
                                        len(SUBJ_ASD) - 1, len(SUBJ_NT) - 1)

T_obs, clusters, cluster_p_values, H0 = clu =\
    spatio_temporal_cluster_test([X1, X2], connectivity=connectivity, n_jobs=1,
                                 threshold=f_threshold)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

fsave_vertices = [np.arange(10242), np.arange(10242)]
tstep = stc_fsaverage_slow.tstep
stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep,
                                             vertices=fsave_vertices,
                                             subject='fsaverage5')

stc_all_cluster_vis.save(savepath + subject + '/' + subject + 'clusters_2_40')