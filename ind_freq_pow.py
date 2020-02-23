import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', '0357',
            '0380', '0381', '0382', '0383'] 

SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0384']

SUBJ = SUBJ_ASD + SUBJ_NT

list_subj = pd.DataFrame(SUBJ, columns=['subj'])
group = pd.DataFrame({'group' : np.concatenate([np.repeat('ASD', 24), np.repeat('NT', 19)])})

post_mag = '/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/individual_alpha_post_mag.csv'
isi_mag = '/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/individual_alpha_isi_mag.csv'
post_grad = '/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/individual_alpha_post_grad.csv'
isi_grad = '/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/individual_alpha_isi_grad.csv'

post_mag1 = '/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/individual_alpha_post_mag_cluster.csv'
isi_mag1 = '/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/individual_alpha_isi_mag_cluster.csv'
post_grad1 = '/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/individual_alpha_post_grad_cluster.csv'
isi_grad1 = '/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/individual_alpha_isi_grad_cluster.csv'
 
data_post_mag = pd.read_csv(post_mag, names=['slow_freq', 'medium_freq', 'fast_freq', 'slow_pow', 'medium_pow', 'fast_pow'])
data_isi_mag = pd.read_csv(isi_mag, names=['slow_freq', 'medium_freq', 'fast_freq', 'slow_pow', 'medium_pow', 'fast_pow'])
data_post_grad = pd.read_csv(post_grad, names=['slow_freq', 'medium_freq', 'fast_freq', 'slow_pow', 'medium_pow', 'fast_pow'])
data_isi_grad = pd.read_csv(isi_grad, names=['slow_freq', 'medium_freq', 'fast_freq', 'slow_pow', 'medium_pow', 'fast_pow'])

data_post_mag1 = pd.read_csv(post_mag1, names=['cluster_slow_pow', 'cluster_medium_pow', 'cluster_fast_pow'])
data_isi_mag1 = pd.read_csv(isi_mag1, names=['cluster_slow_pow', 'cluster_medium_pow', 'cluster_fast_pow'])
data_post_grad1 = pd.read_csv(post_grad1, names=['cluster_slow_pow', 'cluster_medium_pow', 'cluster_fast_pow'])
data_isi_grad1 = pd.read_csv(isi_grad1, names=['cluster_slow_pow', 'cluster_medium_pow', 'cluster_fast_pow'])

frames = [list_subj, group, data_post_mag, data_post_mag1]
result1 = pd.concat(frames, axis=1, sort=False)
result1['cluster_fast_pow'].map(lambda x: '{:.2e}' % x)
result1['cluster_fast_pow']*
frames = [list_subj, group, data_isi_mag, data_isi_mag1]
result2 = pd.concat(frames, axis=1, sort=False)

frames = [list_subj, group, data_post_grad, data_post_grad1]
result3 = pd.concat(frames, axis=1, sort=False)

frames = [list_subj, group, data_isi_grad, data_isi_grad1]
result4 = pd.concat(frames, axis=1, sort=False)

result1.to_excel("/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/data_post_mag.xlsx") 
result2.to_excel("/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/data_isi_mag.xlsx") 
result3.to_excel("/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/data_post_grad.xlsx") 
result4.to_excel("/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/data_isi_grad.xlsx")  

