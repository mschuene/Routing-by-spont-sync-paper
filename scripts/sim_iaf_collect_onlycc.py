import numpy as np
import matplotlib.pyplot as plt
#from utils.spectral import *
from itertools import product
import pickle

proj_name = 'ehe_flicker_sync_better_params_only_cc_tau0'

N =80# 5#80
t = 60#10#5
ra = 1#13
ar = 1#3
ccflickA4 = np.zeros((N,ra,ar,t))
idx_param_array = [(c1,c2,radv,trial)
                   for c1 in np.arange(N)
                   for c2 in np.arange(ra)
                   for radv in np.arange(ar)
                   for trial in range(t)]

ccfA4_leak = np.zeros((N,ra,ar,t))+np.nan
ccfB4_leak = np.zeros((N,ra,ar,t))+np.nan
scfA4_leak = np.zeros((N,ra,ar,t))+np.nan
scfB4_leak = np.zeros((N,ra,ar,t))+np.nan
scfA4_leak_tau0 = np.zeros((N,ra,ar,t,100))+np.nan
scfB4_leak_tau0 = np.zeros((N,ra,ar,t,100))+np.nan




for i,(c1_idx,c2_idx,c3_idx,c4_idx) in enumerate(idx_param_array):
        print(i,flush=True)
        try:
            res = pickle.load(open('/0/maik/attmod/'+proj_name+'/res'+str(i+1)+'.pickle','rb'))
            [ccfAV4_leak,ccfBV4_leak] = res['correlations leak']
            [c4fA_leak,c4fB_leak] = res['spectral coherences leak']
            [c4fA_leak_tau0,c4fB_leak_tau0] = res['spectral coherences leak tau=0']

            ccfA4_leak[c1_idx,c2_idx,c3_idx,c4_idx] = ccfAV4_leak            
            ccfB4_leak[c1_idx,c2_idx,c3_idx,c4_idx] = ccfBV4_leak
            scfA4_leak[c1_idx,c2_idx,c3_idx,c4_idx] = np.sum(np.abs(c4fA_leak))
            scfB4_leak[c1_idx,c2_idx,c3_idx,c4_idx] = np.sum(np.abs(c4fB_leak))
            scfA4_leak_tau0[c1_idx,c2_idx,c3_idx,c4_idx,:] = c4fA_leak_tau0
            scfB4_leak_tau0[c1_idx,c2_idx,c3_idx,c4_idx,:] = c4fB_leak_tau0


        except:
                print('exception',i)
                ccfA4_leak[c1_idx,c2_idx,c3_idx,c4_idx] = np.nan
                ccfB4_leak[c1_idx,c2_idx,c3_idx,c4_idx] = np.nan
                scfA4_leak[c1_idx,c2_idx,c3_idx,c4_idx] = np.nan
                scfB4_leak[c1_idx,c2_idx,c3_idx,c4_idx] = np.nan
                pass




np.savez('/0/maik/attmod/'+proj_name+'/sc_collected_'+proj_name+'.npz',         
         ccfA4_leak=ccfA4_leak, ccfB4_leak=ccfB4_leak,scfA4_leak=scfA4_leak, scfB4_leak=scfB4_leak,
         scfA4_leak_tau0=scfA4_leak_tau0,scfB4_leak_tau0=scfB4_leak_tau0)

