# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:49:25 2019

@author: Girish
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath(__file__)))

def save_obj(obj, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
data1=load_obj('did_sys_enter_AF_2D_2')
data2=load_obj('did_sys_enter_AF_2D_3')
keys_1=list(data1)
indices_1=[(int(keys_1[i][0]*20), int(keys_1[i][1]*20)) for i in range(len(keys_1))]
keys_2=list(data2)
indices_2=[(int(keys_2[i][0]*20), int(keys_2[i][1]*20)) for i in range(len(keys_2))]

phase_space=np.zeros((21,21))

for i in range(len(keys_1)):
    phase_space[indices_1[i][0], indices_1[i][1]]=len(np.argwhere(np.array(data1[keys_1[i]])==True))

for j in range(len(keys_2)):
    phase_space[indices_2[j][0], indices_2[j][1]]=len(np.argwhere(np.array(data2[keys_2[j]])==True))

#phase_space[np.nonzero(phase_space == 0)] = np.NaN
#save_obj(phase_space, 'phase_space')


fig, ax = plt.subplots()
mat = ax.imshow(phase_space, cmap='Blues', vmax=50, interpolation='none', origin='lower', extent=[0,1,0,1])
plt.title(f'No. of times system (L=100) entered AF \n 50 realisations of 50000 runs for each nu_x, nu_y')
plt.plot(np.arange(0,1.01,0.05), 1-np.arange(0,1.01,0.05),'r')
plt.xlabel(r'$\nu_x$')
plt.ylabel(r'$\nu_y$')
plt.xlim(0.2,1)
plt.ylim(0.2,1)
fig.colorbar(mat)