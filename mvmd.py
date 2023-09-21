def mvmd(signal, alpha, tau, K, DC, init, tol):
    """
    % Input and Parameters:
    % ---------------------
    % signal  - input multivariate signal that needs to be decomposed  dim_0: channel dim_1: length
    % alpha   - the parameter that defines the bandwidth of extracted modes (low value of alpha yields higher bandwidth)
    % tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    % K       - the number of modes to be recovered
    % DC      - true if the first mode is put and kept at DC (0-freq)
    % init    - 0 = all omegas start at 0
    %         - 1 = all omegas start uniformly distributed
    %         - 2 = all omegas initialized randomly
    % tol     - tolerance value for convergence of ADMM
    % Output:
    % ----------------------
    % u       - the collection of decomposed modes
    % u_hat   - spectra of the modes
    % omega   - estimated mode center-frequencies

    % Syntax:
    % -----------------------
    % [u, u_hat, omega] = MVMD(X, alpha, tau, K, DC, init, tol)
    %   returns:
    %			 a 3D matrix 'u(K,L,C)' containing K multivariate modes, each with 'C' number of channels and length 'L', that are 
    %            computed by applying the MVMD algorithm on the C-variate signal (time-series) X of length L.
    %    		 - To extract a particular channel 'c' corresponding to all extracted modes, you can use u_c = u(:,:,c).
    %			 - To extract a particular mode 'k' corresponding to all channels, you can use u_k = u(k,:,:).
    %			 - To extract a particular mode 'k' corresponding to the channel 'c', you can use u_kc = u(k,:,c).
    %			 3D matrix 'u_hat(K,L,C)' containing K multivariate modes, each with 'C' number of channels and length 'L', that  
    %            are computed by applying the MVMD algorithm on the C-variate signal (time-series) X of length L.
    %			 2D matrix 'omega(N,K)' estimated mode center frequencies

    [1] N. Rehman, H. Aftab, Multivariate Variational Mode Decomposition, arXiv:1907.04509, 2019. 
    [2] K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Transactions on Signal Processing, vol. 62, pp. 531-544, 2014.
    
    translate from Matlab https://www.mathworks.com/matlabcentral/fileexchange/72814-multivariate-variational-mode-decomposition-mvmd
    Thank you very much!

    translater: Zhao tong
    Contact Email: 1535843978@qq.com
    """
    
    import math
    import numpy as np
    import torch


    C, T = signal.shape # T:length of signal C:  channel number
    # sampling frequency
    fs = 1 / float(T)
    # mirroring
    f = torch.zeros(C, 2 * T)
    f[:,0:T//2] = torch.flip(signal[:,0:T//2], dims=[-1])
    f[:,T//2:3*T//2] = signal
    f[:,3*T//2:2*T] = torch.flip(signal[:,T//2:], dims=[-1])
    # Time Domain 0 to T (of mirrored signal)
    T = float(f.shape[1])  # T = 2 * T
    t = torch.linspace(1/float(T), 1, int(T)) # [T]

    # Spectral Domain discretization
    freqs = t-0.5-1/T
    # Construct and center f_hat
    f_hat = torch.fft.fftshift(torch.fft.fft(f, dim=-1), dim=-1) 
    f_hat_plus = f_hat
    f_hat_plus[:, 0:int(int(T)/2)] = 0

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = 500
    # For future generalizations: individual alpha for each mode
    Alpha=alpha * torch.ones(1, K, dtype=torch.cfloat)

    # matrix keeping track of every iterant 
    u_hat_plus_00 = torch.zeros(len(freqs), C, K, dtype=torch.cfloat)
    u_hat_plus = torch.zeros(len(freqs), C, K, dtype=torch.cfloat)
    omega_plus = torch.zeros(N, K, dtype=torch.cfloat)

    # initialize omegas uniformly
    if (init == 1):
        for i in range(1,K+1):
            omega_plus[0,i-1]=(0.5/K)*(i-1)
    elif (init == 2):
        omega_plus[0,:]=torch.sort(math.exp(math.log(fs))+
            (math.log(0.5)-math.log(fs))*torch.randn(1,K))
    else:
        omega_plus[0,:] = 0
    
    if (DC):
        omega_plus[0,0]=0
    
    # lambda_hat = torch.zeros(len(freqs), C, N)

    # start with empty dual variables
    lambda_hat = torch.zeros(len(freqs), C, N, dtype=torch.cfloat)
    # other inits
    uDiff = tol + 2.2204e-16
    n = 1 # loop counter
    sum_uk = torch.zeros(len(freqs), C) # accumulator

    T = int(T)

    # --------------- Algorithm of MVMD
    while (uDiff > tol and n < N):
        # update modes
        for k in range(K):
            if k > 0:
                sum_uk = u_hat_plus[:,:,k-1] + sum_uk - u_hat_plus_00[:,:,k]
            # else:  # k == 0
            #     sum_uk = u_hat_plus_00[:,:,K-1] + sum_uk - u_hat_plus_00[:,:,k]
            # update spectrum of mode through Wiener filter of residuals
            for c in range(C):
                u_hat_plus[:,c,k] = (f_hat_plus[c,:] - sum_uk[:,c] - \
            lambda_hat[:,c,n-1]/2) /  \
            (1+Alpha[0,k]* torch.pow((freqs - omega_plus[n-1,k]), 2))

            if (DC or (k > 0)):
                # center frequencies  
                numerator = torch.mm(freqs[T//2:T].reshape(1,T//2), 
                            (torch.pow((u_hat_plus[T//2:T,:, k]).abs(),2)))
                # 对列向量求和
                denominator = torch.sum(torch.pow((u_hat_plus[T//2:T,:,k]).abs(),2), dim=0)
                temp1 = torch.sum(numerator)
                temp2 = torch.sum(denominator)
                omega_plus[n,k] = temp1 / temp2
            
        # Dual ascent
        lambda_hat[:,:,n] = lambda_hat[:,:,n-1] + tau * (torch.sum(u_hat_plus,dim=-1) - f_hat_plus.T)
        # loop counter
        n = n + 1
        u_hat_plus_m1 = u_hat_plus_00
        u_hat_plus_00 = u_hat_plus
        # converged yet?
        uDiff = u_hat_plus_00 - u_hat_plus_m1
        uDiff = 1 / T * (uDiff) * uDiff.conj() # conj
        uDiff = 2.2204e-16 + (torch.sum(uDiff[:])).abs()

    # ------ Post-processing and cleanup
    # % discard empty space if converged early
    N = min(N,n)
    omega = omega_plus[0:N,:]
    # Signal reconstruction
    # T = int(T)
    u_hat = torch.zeros(T, K, C, dtype=torch.cfloat)
    for c in range(C):
        u_hat[(T//2):T,:,c] = u_hat_plus[(T//2):T,c,:]
        second_index = list(range(1,T//2+1))
        second_index.reverse()
        u_hat[second_index,:,c] = (u_hat_plus[(T//2):T,c,:]).conj()
        u_hat[0,:,c] = (u_hat[-1,:,c]).conj()

    u = torch.zeros(K,len(t),C, dtype=torch.cfloat)
    for k in range(K):
        for c in range(C):
            u[k,:,c] = (torch.fft.ifft(torch.fft.ifftshift(u_hat[:,k,c]))).real
    
    # remove mirror part
    u = u[:,T//4:3*T//4,:]
    # recompute spectrum
    u_hat = torch.zeros(T//2, K, C, dtype=torch.cfloat)

    for k in range(K):
        for c in range(C):
            u_hat[:,k,c] = torch.fft.fftshift(torch.fft.fft(u[k,:,c])) # .reshape(1,-1).T

    u_hat = torch.permute(u_hat, (1, 0, 2))
    return u, u_hat, omega


"""
import torch
f_1 = 2.0
f_2 = 24.0
f_3 = 288.0
T=1000
t = torch.linspace(1/T,1,T)
v_1 =torch.cos(2*torch.pi*f_1*t)
v_2 = 1/4.0*torch.cos(2*torch.pi*f_2*t)
v_3 = 1/16.0*torch.cos(2*torch.pi*f_3*t)
v = v_1 + v_2 + v_3
v = v.reshape(1,1000)

alpha = 2000
tau = 0
K = 3
DC = 0
init  = 1
tol = 1e-7

[u, u_hat, omega] = mvmd(v, 2000, 0, 3, 0, 1, 1e-7)


import matplotlib.pyplot as plt

a = u[0].reshape(1000)
a = a.numpy()
plt.plot(a)

import torch
T = 1000 
t = torch.linspace(1/T,1,T)
f_channel1 = (torch.cos(2*torch.pi*2*t)) + \
             (1/16*(torch.cos(2*torch.pi*36*t)))
f_channel2 = (1/4*(torch.cos(2*torch.pi*24*t))) + \
             (1/16*(torch.cos(2*torch.pi*36*t)))
f = torch.stack((f_channel1,f_channel2), dim=0) 
"""