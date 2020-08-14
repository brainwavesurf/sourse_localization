#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:14:49 2020

@author: a_shishkina
"""

import mne
from mne.minimum_norm import read_inverse_operator
from mne import io


import numpy as np
import scipy.io

import timeit

start = timeit.default_timer()

datapath = '/net/server/data/Archive/aut_gamma/orekhova/KI/'
savepath = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/Results_Alpha_and_Gamma/'
subjects_dir = datapath + 'freesurfersubjects'

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
 
    fname_raw = datapath + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw.fif'
    fname_inv = savepath + subject + '/' + subject + '_inv'
    
    # Load Data
    raw = io.Raw(fname_raw, preload=True)
    raw.filter(2, 40)
    inverse_operator = read_inverse_operator(fname_inv, verbose=None)
    events = mne.find_events(raw, stim_channel='STI101', verbose=True, shortest_event=1)
    
    # Make epochs from raw    
    
    delay = 8 
    events[:,0] = events[:,0]+delay/1000.*raw.info['sfreq']
    ev = np.sort(np.concatenate((np.where(events[:,2]==2), np.where(events[:,2]==4), np.where(events[:,2]==8)), axis=1))  
    relevantevents = events[ev,:][0]
    
    # Extract epochs from raw
    events_id = dict(V1=2, V2=4, V3=8)
    presim_sec = -1.
    poststim_sec = 1.4
    
    # Load info about preceding events
    info_mat = scipy.io.loadmat(datapath + 'Results_Alpha_and_Gamma/'+ subject + '/' + subject + '_info.mat')
    good_epo = info_mat['ALLINFO']['ep_order_num'][0][0][0]-1
    info_file = info_mat['ALLINFO']['stim_type_previous_tr'][0][0][0]
    goodevents = relevantevents[good_epo]
    goodevents[:,2] = info_file
    
    # Define epochs 
    allepochs = mne.Epochs(raw, goodevents, events_id, tmin=presim_sec, tmax=poststim_sec, baseline=(None, 0), proj=False, preload=True)
    
    # Resample epochs
    if allepochs.info['sfreq']>500:
        allepochs.resample(500)
        
    # Sort epochs according to experimental conditions in the post-stimulus interval
    allepochs.crop(tmin=-0.8, tmax=0)

    allepochs.save(savepath + subject + '/' + subject + '_isi-epo.fif')

stop = timeit.default_timer()
time = stop-start