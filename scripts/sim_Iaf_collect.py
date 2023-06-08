import numpy as np
import matplotlib.pyplot as plt

from itertools import product
import pickle

N =20# 5#80
t = 15#10#5
ra = 1#13
ar = 1#2#3
ccflickA4 = np.zeros((N,ra,ar,t))
idx_param_array = [(c1,c2,radv,trial)
                   for c1 in np.arange(N)
                   for c2 in np.arange(ra)
                   for radv in np.arange(ar)
                   for trial in range(t)]

proj_name = 'ehe_flicker_sync_new_c3_thresh5'#'ehe_flicker_sync_new_c3_thresh1'#iaf_better_params_more_data'

r_A = np.zeros((N,ra,ar,t))+np.nan
r_B = np.zeros((N,ra,ar,t))+np.nan
r_4 = np.zeros((N,ra,ar,t))+np.nan

c_ccAV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccBV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccbAV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccbBV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccfinpAV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccfinpBV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccbfAV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccbfBV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccfAV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccfBV4 = np.zeros((N,ra,ar,t))+np.nan
c_ccfAA = np.zeros((N,ra,ar,t))+np.nan
c_ccfBB = np.zeros((N,ra,ar,t))+np.nan

sc4A = np.zeros((N,ra,ar,t,100))+np.nan
sc4B = np.zeros((N,ra,ar,t,100))+np.nan
sc4flickA = np.zeros((N,ra,ar,t,100))+np.nan
sc4flickB = np.zeros((N,ra,ar,t,100))+np.nan
sc4fA = np.zeros((N,ra,ar,t,100))+np.nan
sc4fB = np.zeros((N,ra,ar,t,100))+np.nan
scfAA = np.zeros((N,ra,ar,t,100))+np.nan
scfBB = np.zeros((N,ra,ar,t,100))+np.nan
scflickAA = np.zeros((N,ra,ar,t,100))+np.nan
scflickBB = np.zeros((N,ra,ar,t,100))+np.nan


for i,(c1_idx,c2_idx,c3_idx,c4_idx) in enumerate(idx_param_array):
        #print(i,flush=True)
        try:
            res = pickle.load(open('/0/maik/attmod/'+proj_name+'/res'+str(i+1)+'.pickle','rb'))

            [pop_act_A,pop_act_B,pop_act_V4,_,_] = res['pop_acts']
            r_A[c1_idx,c2_idx,c3_idx,c4_idx] = pop_act_A
            r_B[c1_idx,c2_idx,c3_idx,c4_idx] = pop_act_B
            r_4[c1_idx,c2_idx,c3_idx,c4_idx] = pop_act_V4
            [ccAV4,ccBV4,ccbAV4,ccbBV4,ccfinpAV4,ccfinpBV4,ccbfAV4,ccbfBV4,ccfAV4,ccfBV4,ccfAA,ccfBB] = res['correlations']
            c_ccAV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccAV4
            c_ccBV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccBV4
            c_ccbAV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccbAV4
            c_ccbBV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccbBV4
            c_ccfinpAV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccfinpAV4
            c_ccfinpBV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccfinpBV4
            c_ccbfAV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccbfAV4
            c_ccbfBV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccbfBV4
            c_ccfAV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccfAV4
            c_ccfBV4[c1_idx,c2_idx,c3_idx,c4_idx] = ccfBV4
            c_ccfAA[c1_idx,c2_idx,c3_idx,c4_idx] = ccfAA
            c_ccfBB[c1_idx,c2_idx,c3_idx,c4_idx] = ccfBB

            [c4A,c4B,c4flickA,c4flickB,c4fA,c4fB,cfAA,cfBB,cflickAA,cflickBB] = res['spectral coherences tau=0']
            sc4A[c1_idx,c2_idx,c3_idx,c4_idx] = c4A
            sc4B[c1_idx,c2_idx,c3_idx,c4_idx] = c4B
            sc4flickA[c1_idx,c2_idx,c3_idx,c4_idx] = c4flickA
            sc4flickB[c1_idx,c2_idx,c3_idx,c4_idx] = c4flickB
            sc4fA[c1_idx,c2_idx,c3_idx,c4_idx] = c4fA
            sc4fB[c1_idx,c2_idx,c3_idx,c4_idx] = c4fB
            scfAA[c1_idx,c2_idx,c3_idx,c4_idx] = cfAA
            scfBB[c1_idx,c2_idx,c3_idx,c4_idx] = cfBB
            scflickAA[c1_idx,c2_idx,c3_idx,c4_idx] = cflickAA
            scflickBB[c1_idx,c2_idx,c3_idx,c4_idx] = cflickBB
        except:
            print('exception',i,flush=True)
            pass
            
            
np.savez('/0/maik/attmod/'+proj_name+'/sc_collected_'+proj_name+'.npz',
         r_A=r_A,
         r_B=r_B,
         r_4=r_4,
         c_ccAV4=c_ccAV4,         
         c_ccBV4=c_ccBV4,
         c_ccbAV4=c_ccbAV4,
         c_ccbBV4=c_ccbBV4,
         c_ccfinpAV4=c_ccfinpAV4,
         c_ccfinpBV4=c_ccfinpBV4,
         c_ccbfAV4=c_ccbfAV4,
         c_ccbfBV4=c_ccbfBV4,
         c_ccfAV4=c_ccfAV4,
         c_ccfBV4=c_ccfBV4,
         c_ccfAA=c_ccfAA,
         c_ccfBB=c_ccfBB,
         sc4A=sc4A,
         sc4B=sc4B,
         sc4flickA=sc4flickA,
         sc4flickB=sc4flickB,
         sc4fA=sc4fA,
         sc4fB=sc4fB,
         scfAA=scfAA,
         scfBB=scfBB,
         scflickAA=scflickAA,
         scflickBB=scflickBB)
