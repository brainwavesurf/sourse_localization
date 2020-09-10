'''Case 1:
    granger_all_v1_mt = granger_all_mt_v1
'''
granger_v1_mt_avg = granger_mt_v1_avg = np.empty((1,203,nvert_v1)) #(1,203,111)
    for idx_v1 in range(nvert_v1): 
        granger_v1_mt = granger_mt_v1 = np.empty((1,203,nvert_mt)) #(1,203,26)
        for idx_mt in range(nvert_mt):
            # Create input to connectivity analysis from two different timecourse (times,epochs,signals)
            signal_v1_mt = np.append(signal_v1[:,:,idx_v1,np.newaxis], signal_mt[:,:,idx_mt,np.newaxis], axis=2)
            signal_mt_v1 = np.append(signal_mt[:,:,idx_mt,np.newaxis], signal_v1[:,:,idx_v1,np.newaxis], axis=2)
            # Compute granger causality
            m_v1_mt = Multitaper(signal_v1_mt, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            m_mt_v1 = Multitaper(signal_mt_v1, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            c_v1_mt = Connectivity(fourier_coefficients=m_v1_mt.fft(), frequencies=m_v1_mt.frequencies)
            c_mt_v1 = Connectivity(fourier_coefficients=m_mt_v1.fft(), frequencies=m_mt_v1.frequencies)
            granger_v1_mt[:,:,idx_mt] = c_v1_mt.pairwise_spectral_granger_prediction()[...,0,1]
            granger_mt_v1[:,:,idx_mt] = c_mt_v1.pairwise_spectral_granger_prediction()[...,0,1]
        # Average GC values computed for one V1 vertex and all mt vertices     
        granger_v1_mt_avg[:,:,idx_v1] = granger_v1_mt.mean(2)
        granger_mt_v1_avg[:,:,idx_v1] = granger_mt_v1.mean(2)
    # Average GC values computed for all V1 vertex and all mt vertices  
    granger_all_v1_mt = granger_v1_mt_avg.mean(2) # (1,203) , where 203 is freqs
    granger_all_mt_v1 = granger_mt_v1_avg.mean(2)

'''Case 1:
    granger_all_v1_mt != granger_all_mt_v1
'''

granger_v1_mt_avg = np.empty((1,203,nvert_v1)) #(1,203,111)
    for idx_v1 in range(nvert_v1): 
        granger_v1_mt = np.empty((1,203,nvert_mt)) #(1,203,26)
        for idx_mt in range(nvert_mt):
            # Create input to connectivity analysis from two different timecourse (times,epochs,signals)
            signal_v1_mt = np.append(signal_v1[:,:,idx_v1,np.newaxis], signal_mt[:,:,idx_mt,np.newaxis], axis=2)
            # Compute granger causality
            m_v1_mt = Multitaper(signal_v1_mt, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            c_v1_mt = Connectivity(fourier_coefficients=m_v1_mt.fft(), frequencies=m_v1_mt.frequencies)
            granger_v1_mt[:,:,idx_mt] = c_v1_mt.pairwise_spectral_granger_prediction()[...,0,1]
        # Average GC values computed for one V1 vertex and all mt vertices     
        granger_v1_mt_avg[:,:,idx_v1] = granger_v1_mt.mean(2)
    # Average GC values computed for all V1 vertex and all mt vertices  
    granger_all_v1_mt = granger_v1_mt_avg.mean(2) # (1,203) , where 203 is freqs
    
    
granger_mt_v1_avg = np.empty((1,203,nvert_v1)) #(1,203,111)
    for idx_v1 in range(nvert_v1): 
        granger_mt_v1 = np.empty((1,203,nvert_mt)) #(1,203,26)
        for idx_mt in range(nvert_mt):
            # Create input to connectivity analysis from two different timecourse (times,epochs,signals)
            signal_mt_v1 = np.append(signal_mt[:,:,idx_mt,np.newaxis], signal_v1[:,:,idx_v1,np.newaxis], axis=2)
            # Compute granger causality
            m_mt_v1 = Multitaper(signal_mt_v1, sfreq, time_halfbandwidth_product=2, start_time=-0.8, n_tapers=1)
            c_mt_v1 = Connectivity(fourier_coefficients=m_mt_v1.fft(), frequencies=m_mt_v1.frequencies)
            granger_mt_v1[:,:,idx_mt] = c_mt_v1.pairwise_spectral_granger_prediction()[...,0,1]
        # Average GC values computed for one V1 vertex and all mt vertices     
        granger_mt_v1_avg[:,:,idx_v1] = granger_mt_v1.mean(2)
    # Average GC values computed for all V1 vertex and all mt vertices  
    granger_all_mt_v1 = granger_mt_v1_avg.mean(2) # (1,203) , where 203 is freqs
    
    