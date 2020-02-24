#import packages
import mne
from mne import io
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import numpy as np
from mne.time_frequency import psd_array_welch
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs


#load subj info
SUBJ_NT = ['0101', '0102', '0103', '0104', '0105', '0136', '0137', '0138',
           '0140', '0158', '0162', '0163', '0178', '0179', '0255', '0257', '0348', 
           '0378', '0379', '0384']
                       
SUBJ_ASD = ['0106', '0107', '0139', '0141', '0159', '0160', '0161',  
            '0164', '0253', '0254', '0256', '0273', '0274', '0275',
            '0276', '0346', '0347', '0351', '0358', 
            '0380', '0381', '0382', '0383'] 
SUBJECTS = SUBJ_ASD + SUBJ_NT
SUBJECTS = ['0106']
PATHfrom = '/home/a_shishkina/data/KI/'
subjects_dir = PATHfrom + 'freesurfersubjects'
for subject in SUBJECTS:
    subjpath = PATHfrom  + 'SUBJECTS/' + subject + '/ICA_nonotch_crop'+ '/epochs/'
    savepath = PATHfrom + 'Results_Alpha_and_Gamma/'

    epoch_type = '-lagcorrected-epo.fif'
    #create bem model and make its solution
    conductivity = (0.3,)  # for single layer
    model = mne.make_bem_model(subject='Case'+subject, ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    raw_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw.fif'
    
    raw = io.Raw(raw_fname, preload=True)
    picks = mne.pick_types(raw.info,  meg='planar1')
    #set up a source space (inflated brain); if volume - use pos=5
    src = mne.setup_source_space('Case'+subject, spacing='oct6',
                                subjects_dir=subjects_dir, add_dist=False)
    
    trans = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/' + subject + '_rings_ICA_raw-trans.fif'
    
    #make forward solution
    fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.0, n_jobs=2)
    
    original_data = mne.io.read_raw_fif(raw_fname, preload=False)
    original_info = original_data.info
    
    ftname = savepath + subject + '/' + subject + '_fieldtrip_epochs.mat'
    
    #load csp data for CSP1 fast
    csp1_fast1_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='epochs_fast1', trialinfo_column=0)
    csp1_fast1_epo.save('/home/a_shishkina/data/KI/SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_csp_epo_fast1.fif')
    csp_fast_fname = '/home/a_shishkina/data/KI/SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_csp_epo_fast1.fif'
    fast_csp1_epo = mne.read_epochs(csp_fast_fname, proj=False, verbose=None) 
    
    #load csp data for CSP1 slow
    csp1_slow1_epo = mne.read_epochs_fieldtrip(ftname, original_info, data_name='epochs_slow1', trialinfo_column=0)
    csp1_slow1_epo.save('/home/a_shishkina/data/KI/SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_csp_epo_slow1.fif')
    csp_slow_fname = '/home/a_shishkina/data/KI/SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_csp_epo_slow1.fif'
    slow_csp1_epo = mne.read_epochs(csp_slow_fname, proj=False, verbose=None) 
    
    #load bandpassed for fast poststim
    filename = subjpath + subject + '_preproc_alpha_bp_epochs.mat'
    epo_fast = mne.read_epochs_fieldtrip(filename, original_info, data_name='fast_alpha_post', trialinfo_column=0)
    epo_fast.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif')
    epo_fast_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_fast.fif'
    fast_epo = mne.read_epochs(epo_fast_fname, proj=False, verbose=None) 
    
    #load bandpassed for fast poststim
    filename = subjpath + subject + '_preproc_alpha_bp_epochs.mat'
    epo_slow = mne.read_epochs_fieldtrip(filename, original_info, data_name='slow_alpha_post', trialinfo_column=0)
    epo_slow.save(PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif')
    epo_slow_fname = PATHfrom + 'SUBJECTS/' + subject + '/ICA_nonotch_crop/epochs/' + subject + '_epo_slow.fif'
    slow_epo = mne.read_epochs(epo_slow_fname, proj=False, verbose=None)

    noise_cov_fast = mne.compute_covariance(fast_epo, method='shrinkage', rank=None)
    noise_cov_slow = mne.compute_covariance(slow_epo, method='shrinkage', rank=None)
    data_cov_fast = mne.compute_covariance(fast_csp1_epo, method='shrinkage', rank=None)
    data_cov_slow = mne.compute_covariance(slow_csp1_epo, method='shrinkage', rank=None)
    
    #fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov_fast, raw.info)
    #fig_cov, fig_spectra = mne.viz.plot_cov(data_cov_slow, raw.info)
    
    #provide analysis for selected labels
    labs = mne.read_labels_from_annot('Case'+subject, parc='aparc', subjects_dir=subjects_dir)
    labels = [labs[1], labs[6], labs[7], labs[12], labs[13], labs[14], labs[15], labs[22], labs[23], labs[26], labs[27], labs[42], labs[43], labs[50], labs[51], labs[58], labs[59]]
    
    #create big label with all selected labels
    sum_labels=labs[0]
    for elem in labels:
        sum_labels = sum_labels + elem

    filters_csp1_fast = make_lcmv(fast_csp1_epo.info, fwd, data_cov_fast, reg=0.05,
                        noise_cov=noise_cov_fast, pick_ori='max-power',
                        weight_norm='nai', rank=None, label=sum_labels)
    
    filters_csp1_slow = make_lcmv(slow_csp1_epo.info, fwd, data_cov_slow, reg=0.05,
                        noise_cov=noise_cov_slow, pick_ori='max-power',
                        weight_norm='nai', rank=None, label=sum_labels)
    
    #sourse estimates on each epoch
    stc_csp1_fast = apply_lcmv_epochs(fast_csp1_epo, filters_csp1_fast, max_ori_out='signed')
    stc_csp1_slow = apply_lcmv_epochs(slow_csp1_epo, filters_csp1_slow, max_ori_out='signed')

    stc_csp1_fast_av = np.mean(stc_csp1_fast)
    stc_csp1_slow_av = np.mean(stc_csp1_slow)
    
    V_csp_fast = []
    V_csp_slow = []
    
    stc_fast = [stc_csp1_fast]
    stc_slow = [stc_csp1_slow]

    for s in stc_fast:
        temp = [element.data for element in s]
        V_csp_fast.append(np.stack(temp))
    for s in stc_slow:    
        temp = [element.data for element in s]
        V_csp_slow.append(np.stack(temp))
    
    #calculate spectral power on each epoch
    psds_fast, freqs = psd_array_welch(V_csp_fast[0], sfreq=500, n_fft=512, n_overlap=0, n_per_seg = 400,fmin=10, fmax=17, n_jobs=1)
    psds_slow, freqs = psd_array_welch(V_csp_slow[0], sfreq=500, n_fft=512, n_overlap=0, n_per_seg = 400,fmin=10, fmax=17, n_jobs=1)
    
    psds_av_fast = psds_fast.mean(axis=0)
    psds_av_slow = psds_slow.mean(axis=0)
    
    #calculate normalized spectral power
    CSP1_power = psds_av_fast - psds_av_slow
    
    maxes_ind = np.argmax(CSP1_power[:,:])
    #find max values among 41-80 Hz and get 26 which come first
    Vmax_voxels_num = np.argsort(CSP1_power[maxes_ind])[-26:] #order numbers of 26 max voxels
    
    #find max values among 41-80 Hz and get 26 which come first
    Vmax_voxels_num.sort() #sorted from min to max
    
    #get 'names' of 26 max vertices that are in lh and in rh
    q = stc_csp1_fast[0]
    Vmax_voxels_num_lh = Vmax_voxels_num[Vmax_voxels_num<len(q.lh_vertno)] #order numbers of 26 max voxels (left hemi)
    Vmax_voxels_num_rh = Vmax_voxels_num[Vmax_voxels_num>=len(q.lh_vertno)] #order numbers of 26 max voxels (right hemi)
    
    #get 'names' of all vertices in lh and rh
    lh_vox_num = stc_csp1_fast_av.lh_vertno #'names' of all left hemi voxels
    rh_vox_num = stc_csp1_fast_av.rh_vertno #'names' of all right hemi voxels
    
    CSP1_power_rh = np.max(CSP1_power[maxes_ind][Vmax_voxels_num_rh], axis=0) #rh data of 26 max voxels
    CSP1_power_lh = np.max(CSP1_power[maxes_ind][Vmax_voxels_num_lh], axis=1) #lh data of 26 max voxels
    
    #combine 'names' of left and right hemi voxels to get names of our 26 max voxels:
    vox_num = np.hstack([lh_vox_num, rh_vox_num])
    #now get 'names' of our 26 max voxels:
    vox_num_26 = vox_num[Vmax_voxels_num] #our 26 'names'
    
    #sourse estimation for the whole brain; lh - first, rh - second
    stc_my = mne.SourceEstimate(CSP1_power, [stc_csp1_fast_av.lh_vertno, stc_csp1_fast_av.rh_vertno], tmin=-0.8, tstep=stc_csp1_fast_av.tstep, subject=subject)
    
    hemis = ['lh', 'rh']
    for he in hemis:
    	brain = stc_my.plot(hemi=he, subjects_dir=subjects_dir,  clim=dict(kind='value', lims=[0, CSP1_power.max()/2, CSP1_power.max()]))
    