# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:35:29 2023

@author: Admin
"""

import torch
import matplotlib.pyplot as plt
from mvmd_python import mvmd

T = 1000 
t = torch.linspace(1/T,1,T)

f_channel1 = (10*torch.cos(2*torch.pi*2*t)) + \
             (9*(torch.cos(2*torch.pi*36*t)))
             
f_channel2 = (9*(torch.cos(2*torch.pi*24*t))) + \
             (8*(torch.cos(2*torch.pi*36*t)))
             
f_channel3 = (8*(torch.cos(2*torch.pi*28*t))) + \
                          (7*(torch.cos(2*torch.pi*48*t)))
                          
f_channel4 = (7*(torch.cos(2*torch.pi*32*t))) + \
                          (6*(torch.cos(2*torch.pi*36*t)))
                          
f_channel5 = (6*(torch.cos(2*torch.pi*19*t))) + \
                (5*(torch.cos(2*torch.pi*64*t)))
                
f_channel6 = (5*(torch.cos(2*torch.pi*17*t))) + \
                (4*(torch.cos(2*torch.pi*13*t)))
                
f_channel7 = (4*(torch.cos(2*torch.pi*27*t))) + \
                (3*(torch.cos(2*torch.pi*13*t)))
                
f_channel8 = (3*(torch.cos(2*torch.pi*56*t))) + \
                (2*(torch.cos(2*torch.pi*20*t)))
                
f_channel9 = (2*(torch.cos(2*torch.pi*48*t))) + \
                (1*(torch.cos(2*torch.pi*20*t)))
                
f = torch.stack((f_channel1,f_channel2,f_channel3,f_channel4,f_channel5,f_channel6,f_channel7,
                 f_channel8, f_channel9),dim=0) 


[u, u_hat, omega] = mvmd(f, 2000, 0, 3, 0, 1, 1e-7, 50)
# u = torch.fft.fftshift(u,dim=-1)

nrows, ncols = 2, f.shape[0]

fig, ax = plt.subplots(nrows, ncols)
for r in range(nrows):
    for c in range(ncols):
        if r == 0:
            # original data
            ax[r][c].plot(f[c].numpy())
        if r == 1:
            # reconstruct
            ax[r][c].plot(torch.sum(u[:,:,c].real,dim=0).numpy())