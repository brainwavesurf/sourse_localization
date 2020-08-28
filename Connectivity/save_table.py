#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:28:11 2020

@author: a_shishkina
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

savepath = '/net/server/data/Archive/aut_gamma/orekhova/KI/Scripts_bkp/Shishkina/KI/Results_Alpha_and_Gamma/'

SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383'] 

SUBJECTS = SUBJ_ASD + SUBJ_NT

# Fast condition

# Load values of WPLI2_debiased values
avg_tc = np.load(savepath + 'wpli2_debiased/' + 'all_v1_mt_avg_fast.npy')
avg_tcs = avg_tc[:,0,:]
all_vert = np.load(savepath + 'wpli2_debiased/' + 'all_v1_mt_vertices_fast_fastepo.npy')
# Load values of spectral power
freq_pow = np.load(savepath + 'pow/' + 'all_fast_slow_v1_mt_freq_pow.npy')
pow_fast_v1 = freq_pow[:,0,1,:]
pow_fast_mt = freq_pow[:,2,1,:]
# Load values of frequency
freqs = freq_pow[1,2,0,:]

# Create DataFrames
df_avg_tcs = pd.DataFrame(avg_tcs[:,0],index=SUBJECTS, columns=["%.2f" %freqs[0]]) 
for s in np.arange(1,31):
    df_avg_tcs["%.2f" %freqs[s]] = avg_tcs[:,s]
    
df_all_vert = pd.DataFrame(all_vert[:,0],index=SUBJECTS, columns=["%.2f" %freqs[0]]) 
for s in np.arange(1,31):
    df_all_vert["%.2f" %freqs[s]] = all_vert[:,s]
    
df_pow_fast_v1 = pd.DataFrame(pow_fast_v1[:,0],index=SUBJECTS, columns=["%.2f" %freqs[0]]) 
for s in np.arange(1,31):
    df_pow_fast_v1["%.2f" %freqs[s]] = pow_fast_v1[:,s]
    
df_pow_fast_mt = pd.DataFrame(pow_fast_mt[:,0],index=SUBJECTS, columns=["%.2f" %freqs[0]]) 
for s in np.arange(1,31):
    df_pow_fast_mt["%.2f" %freqs[s]] = pow_fast_mt[:,s]
    
df_fast = pd.concat([df_avg_tcs, df_all_vert, df_pow_fast_v1, df_pow_fast_mt], axis=1,
                    keys=['avg_tcs','all_vert', 'pow_v1', 'pow_mt'])
# Save
df_fast.to_excel(savepath + 'table/' + 'fastepo.xlsx')


# Slow condition

# Load values of WPLI2_debiased values
avg_tc = np.load(savepath + 'wpli2_debiased/' + 'all_v1_mt_avg_slow.npy')
avg_tcs = avg_tc[:,0,:]
all_vert = np.load(savepath + 'wpli2_debiased/' + 'all_v1_mt_vertices_fast_slowepo.npy')
# Load values of spectral power
freq_pow = np.load(savepath + 'pow/' + 'all_fast_slow_v1_mt_freq_pow.npy')
pow_slow_v1 = freq_pow[:,1,1,:]
pow_slow_mt = freq_pow[:,3,1,:]
# Load values of frequency
freqs = freq_pow[1,2,0,:]

# Create DataFrames
df_avg_tcs = pd.DataFrame(avg_tcs[:,0],index=SUBJECTS, columns=["%.2f" %freqs[0]]) 
for s in np.arange(1,31):
    df_avg_tcs["%.2f" %freqs[s]] = avg_tcs[:,s]
    
df_all_vert = pd.DataFrame(all_vert[:,0],index=SUBJECTS, columns=["%.2f" %freqs[0]]) 
for s in np.arange(1,31):
    df_all_vert["%.2f" %freqs[s]] = all_vert[:,s]
    
df_pow_slow_v1 = pd.DataFrame(pow_slow_v1[:,0],index=SUBJECTS, columns=["%.2f" %freqs[0]]) 
for s in np.arange(1,31):
    df_pow_slow_v1["%.2f" %freqs[s]] = pow_slow_v1[:,s]
    
df_pow_slow_mt = pd.DataFrame(pow_slow_mt[:,0],index=SUBJECTS, columns=["%.2f" %freqs[0]]) 
for s in np.arange(1,31):
    df_pow_slow_mt["%.2f" %freqs[s]] = pow_slow_mt[:,s]
    
df_slow = pd.concat([df_avg_tcs, df_all_vert, df_pow_slow_v1, df_pow_slow_mt], axis=1,
                    keys=['avg_tcs','all_vert', 'pow_v1', 'pow_mt'])
# Save
df_slow.to_excel(savepath + 'table/' + 'slowepo.xlsx')

# Check
plt.plot(freqs, df_avg_tcs.mean(0), label='avg_tcs')
plt.plot(freqs, df_all_vert.mean(0), label='all_vert')
plt.plot(freqs, df_pow_slow_v1.mean(0), label='pow_slow_v1')
plt.plot(freqs, df_pow_slow_mt.mean(0), label='pow_slow_mt')
plt.plot(freqs, df_pow_fast_v1.mean(0), label='pow_fast_v1')
plt.plot(freqs, df_pow_fast_mt.mean(0), label='pow_fast_mt')
plt.legend()





