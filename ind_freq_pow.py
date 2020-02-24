import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

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

data_post_mag = pd.read_csv(post_mag, names=['slow_fw', 'med_fw', 'fast_fw', 'slow_fwo', 'med_fwo', 'fast_fwo', 'slow', 'medium', 'fast', 'slow', 'medium', 'fast', 'slow_pw', 'med_pw', 'fast_pw', 'slow_pwo', 'med_pwo', 'fast_pwo', 'slow_cw', 'med_cw', 'fast_cw', 'slow_cwo', 'med_cwo', 'fast_cwo'])
data_isi_mag = pd.read_csv(isi_mag, names=['slow_fw', 'med_fw', 'fast_fw', 'slow_fwo', 'med_fwo', 'fast_fwo', 'slow', 'medium', 'fast', 'slow', 'medium', 'fast', 'slow_pw', 'med_pw', 'fast_pw', 'slow_pwo', 'med_pwo', 'fast_pwo', 'slow_cw', 'med_cw', 'fast_cw', 'slow_cwo', 'med_cwo', 'fast_cwo'])
data_post_grad = pd.read_csv(post_grad, names=['slow_fw', 'med_fw', 'fast_fw', 'slow_fwo', 'med_fwo', 'fast_fwo', 'slow', 'medium', 'fast', 'slow', 'medium', 'fast', 'slow_pw', 'med_pw', 'fast_pw', 'slow_pwo', 'med_pwo', 'fast_pwo', 'slow_cw', 'med_cw', 'fast_cw', 'slow_cwo', 'med_cwo', 'fast_cwo'])
data_isi_grad = pd.read_csv(isi_grad, names=['slow_fw', 'med_fw', 'fast_fw', 'slow_fwo', 'med_fwo', 'fast_fwo', 'slow', 'medium', 'fast', 'slow', 'medium', 'fast', 'slow_pw', 'med_pw', 'fast_pw', 'slow_pwo', 'med_pwo', 'fast_pwo', 'slow_cw', 'med_cw', 'fast_cw', 'slow_cwo', 'med_cwo', 'fast_cwo'])

frames = [list_subj, group, data_post_mag]
result1 = pd.concat(frames, axis=1, sort=False)

frames = [list_subj, group, data_isi_mag]
result2 = pd.concat(frames, axis=1, sort=False)

frames = [list_subj, group, data_post_grad]
result3 = pd.concat(frames, axis=1, sort=False)

frames = [list_subj, group, data_isi_grad]
result4 = pd.concat(frames, axis=1, sort=False)

result1.to_excel("/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/data_post_mag.xlsx") 
result2.to_excel("/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/data_isi_mag.xlsx") 
result3.to_excel("/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/data_post_grad.xlsx") 
result4.to_excel("/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/data_isi_grad.xlsx")  

#do stat
data = [result1, result2, result3, result4]
for df in data:
    #take only interested columns
    df = pd.DataFrame(df[['slow_fw', 'med_fw', 'fast_fw', 'slow_fwo', 'med_fwo', 'fast_fwo', 'slow_pw', 'med_pw', 'fast_pw', 'slow_pwo', 'med_pwo', 'fast_pwo', 'slow_cw', 'med_cw', 'fast_cw', 'slow_cwo', 'med_cwo', 'fast_cwo']])
   
    #boxplot original data
    fig, axes = plt.subplots(2,3)
    fig.suptitle('Boxplots of all data')
    df[['slow_fw', 'med_fw', 'fast_fw']].plot(kind='box',ax=axes[0,0],title = 'Freq_with')
    df[['slow_fwo', 'med_fwo', 'fast_fwo']].plot(kind='box',ax=axes[0,1],title = 'Freq_without')
    df[['slow_pw', 'med_pw', 'fast_pw']].plot(kind='box',ax=axes[0,2],title = 'Pow_with')
    df[['slow_pwo', 'med_pwo', 'fast_pwo']].plot(kind='box',ax=axes[1,0],title = 'Pow_without')
    df[['slow_cw', 'med_cw', 'fast_cw']].plot(kind='box',ax=axes[1,1],title = 'Pow_clus_with')
    df[['slow_cwo', 'med_cwo', 'fast_cwo']].plot(kind='box',ax=axes[1,2],title = 'Pow_clus_without')
    # save the figure to file
    fig.savefig('/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/boxplots.png')  
    plt.close(fig) 
    
    #check outliers by z-method
    z = np.abs(stats.zscore(df))
    #remove outliers
    df_o = df[(z < 3).all(axis=1)]
    #boxplot data
    fig, axes = plt.subplots(2,3)
    fig.suptitle('Boxplots after z-method')
    df_o[['slow_fw', 'med_fw', 'fast_fw']].plot(kind='box',ax=axes[0,0],title = 'Freq_with')
    df_o[['slow_fwo', 'med_fwo', 'fast_fwo']].plot(kind='box',ax=axes[0,1],title = 'Freq_without')
    df_o[['slow_pw', 'med_pw', 'fast_pw']].plot(kind='box',ax=axes[0,2],title = 'Pow_with')
    df_o[['slow_pwo', 'med_pwo', 'fast_pwo']].plot(kind='box',ax=axes[1,0],title = 'Pow_without')
    df_o[['slow_cw', 'med_cw', 'fast_cw']].plot(kind='box',ax=axes[1,1],title = 'Pow_clus_with')
    df_o[['slow_cwo', 'med_cwo', 'fast_cwo']].plot(kind='box',ax=axes[1,2],title = 'Pow_clus_without')
    # save the figure to file
    fig.savefig('/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/boxplots_after_z.png')  
    plt.close(fig) 
    
    #check outliers by IQR Score
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    #remove outliers
    df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    #boxplot data
    #boxplot data
    fig, axes = plt.subplots(2,3)
    fig.suptitle('Boxplots after IQR-method')
    df_out[['slow_fw', 'med_fw', 'fast_fw']].plot(kind='box',ax=axes[0,0],title = 'Freq_with')
    df_out[['slow_fwo', 'med_fwo', 'fast_fwo']].plot(kind='box',ax=axes[0,1],title = 'Freq_without')
    df_out[['slow_pw', 'med_pw', 'fast_pw']].plot(kind='box',ax=axes[0,2],title = 'Pow_with')
    df_out[['slow_pwo', 'med_pwo', 'fast_pwo']].plot(kind='box',ax=axes[1,0],title = 'Pow_without')
    df_out[['slow_cw', 'med_cw', 'fast_cw']].plot(kind='box',ax=axes[1,1],title = 'Pow_clus_with')
    df_out[['slow_cwo', 'med_cwo', 'fast_cwo']].plot(kind='box',ax=axes[1,2],title = 'Pow_clus_without')
    # save the figure to file
    fig.savefig('/home/a_shishkina/data/KI/Results_Alpha_and_Gamma/1_results/boxplots_after_IQR.png')  
    plt.close(fig) 
    
    df_out['fw_difference'] = df_out['fast_cw'] - df_out['slow_cw']
    df_out['group'] = result1['group']
    df_out['fw_difference'].plot(kind='hist', title= 'Freq Difference Histogram')
    stats.probplot(df_out['fw_difference'], plot= plt)
    stats.shapiro(df_out['fw_difference'])
    stats.ttest_rel(df_out['fast_fw'], df_out['slow_fw'])

    sns.boxplot(x="group", y="fw_difference", palette=["m", "g"], data=df_out)