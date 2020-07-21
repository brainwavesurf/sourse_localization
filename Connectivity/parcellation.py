# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 08:40:31 2020

@author: a_shishkina
"""
import mne

datapath = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
subjects_dir = datapath + 'freesurfersubjects/'

#load subj info
SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383'] 

SUBJECTS = SUBJ_ASD + SUBJ_NT

for subject in SUBJECTS:
    
    labels_lh = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)
    labels_rh = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'rh', subjects_dir=subjects_dir)
    
    #v1
    V1_label_lh = [label for label in labels_lh if label.name == 'L_V1_ROI-lh'][0]
    V1_label_rh = [label for label in labels_rh if label.name == 'R_V1_ROI-rh'][0]
    
    #mt
    MT_label_lh = [label for label in labels_lh if label.name == 'L_MT_ROI-lh'][0]
    MT_label_rh = [label for label in labels_rh if label.name == 'R_MT_ROI-rh'][0]
    #morph labels to subjects
    V1_subj_lh = V1_label_lh.morph('fsaverage', 'Case' + subject, subjects_dir = subjects_dir)
    V1_subj_rh = V1_label_rh.morph('fsaverage', 'Case' + subject, subjects_dir = subjects_dir)
    
    MT_subj_lh = MT_label_lh.morph('fsaverage', 'Case' + subject, subjects_dir = subjects_dir)
    MT_subj_rh = MT_label_rh.morph('fsaverage', 'Case' + subject, subjects_dir = subjects_dir)
    
    savepath = datapath + 'Results_Alpha_and_Gamma/'
    V1_subj_lh.save(savepath + subject + '/' + subject + '_V1_lh')
    V1_subj_rh.save(savepath + subject + '/' + subject + '_V1_rh')
    
    MT_subj_lh.save(savepath + subject + '/' + subject + '_MT_lh')
    MT_subj_rh.save(savepath + subject + '/' + subject + '_MT_rh')
