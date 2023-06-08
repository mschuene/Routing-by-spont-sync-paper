from tqdm import tqdm
import itertools 
from utils.spectral import *
import numba
import pickle
import mpmath as m   


@numba.njit(cache=True)
def sim_iaf(ns,dt,ext_inputs,tau = 10e-3,Vthr = 1,dV = 1/20):
    V = 0#np.zeros((ns,))
    s = []
    for i in range(ns-1):
        V = V+dt*(-V)/tau
        V+= dV*ext_inputs[i]
        if V>Vthr:
            s.append(i)
            V -= 1
            #V = 0
    return V,s

nf = 5
d_levels = np.linspace(-1,1,nf)
def flicker(levels=None):
    levels = d_levels if levels is None else levels
    return levels[np.random.randint(0,len(levels))]



@numba.njit(cache=True)
def get_avinc(A,Ns,nNs):
    pop_ids = lambda k: slice(nNs[k],nNs[k+1],None)
    avinc = np.zeros(len(Ns),dtype=np.int64)
    for k in range(len(Ns)):
        avinc[k]+= np.sum(A[pop_ids(k)])
    return avinc



@numba.njit(cache=True)
def simulate_model_subnets_nonorm_thresholds(Ns,W,p,deltaU,thresh=0,num_steps=None,num_av=None,u0=None,outdir=None):
    th_idx = 4 #TODO Hardcoded index!!!!
    Ns = np.array(Ns,dtype=np.int64)
    N = np.sum(Ns)
    u = np.random.random((N,)) if u0 is None else u0
    avc = 0
    step = 0
    neurons = np.arange(N)
    I = np.eye(len(Ns))
    avs = []
    avt = []
    #pc = np.cumsum(np.concatenate(tuple([np.array([ps/n]*n) for ps,n in zip(p,Ns)])))
    avu = []
    ks = []
    cNs = np.cumsum(Ns)
    nNs = np.concatenate((np.array([0]),cNs))
    pop_ids = lambda k: slice(nNs[k],nNs[k+1],None)
    pbig = np.zeros(N)
    for k in range(len(Ns)):
        pbig[pop_ids(k)] = p[k]/Ns[k]
    pc = np.cumsum(pbig)
    u_saved = u.copy()
    #pc = np.cumsum(np.concatenate([[ps/n]*n for ps,n in zip(p,Ns)]))

    while avc < num_av if num_av is not None else step < num_steps:
        k = np.searchsorted(pc,np.random.random())
        k_sub = np.searchsorted(cNs,k+1)
        ks.append(k_sub)
        u_saved = u.copy()
        u[k] += deltaU
        if u[k] > 1:
            avc+= 1
            avsize = np.zeros(len(Ns),dtype=np.int64)
            A = u > 1;
            avinc = get_avinc(A,Ns,nNs) # vector len(Ns)
            avt.append(step)
            avunits = []
            while np.sum(avinc) > 0:
                avsize += avinc
                u[A] -= 1
                rec_int = W@avinc.astype(np.float64)
                for i in range(len(Ns)):
                    u[pop_ids(i)]+=rec_int[i]#/Ns[i]
                A = u>1
                avinc = get_avinc(A,Ns,nNs)
            if (k_sub != th_idx) and  (avsize[0]+avsize[2]<thresh): # TODO Hardcoded indices!!!!
                u[pop_ids(th_idx)] = u_saved[pop_ids(th_idx)]
                avsize[th_idx] = 0 
            avs.append(avsize)
            #avu.append(avunits)
        step += 1
    return avs,np.array(avt),u,ks




def bin_data(data,bins):
    slices = np.linspace(0, len(data), bins+1, True).astype(np.int)
    counts = np.diff(slices)
    mean = np.add.reduceat(data, slices[:-1]) / counts
    return mean


param_array = [(c1,radv,method,trial)
               for c1 in np.linspace(0.2,1,80)
               for radv in [12]
               for method in ['int']#,'ext']
               for trial in range(60)]

c1,radv,method,trial = param_array[task_id-1]#[0.8,12,'int',0]#param_array[task_id-1]#
#c1,radv,method = 

rate1=40;
cup=0.3;
T=250; # war 150
n=100;
n2 = int(n/10);
c=0.25;
thresh=5; # for n=1000
fname=None#'./res_kadabuum'
dtcorr=3

rate2 = rate1+radv
Ns = (n,n2,n,n2,n,1)
ac = 1-1/np.sqrt(n)
c2 = c1
c3 = c2-2*cup
N = 3*n+2*n2+1


Ws = np.zeros((6,6))#one extra unit to supply the overshoot of deltaU external inputs
Ws[0,0] = c1*ac/n # V1 A
# Ws[1,:] is inhibited second population of nonattended V1
Ws[2,2] = c2*ac/n # V1 B 
Ws[3,3] = (1-np.sqrt(n2)/n2)/n2
if method=='int':
    Ws[2,3] = 0.095/n2 # input from crit subnet to V1 B # 0.03 für n=1000
Ws[4,4] = c3*ac/n # V4 pop
Ws[4,0] = cup/n # inputs to V4
Ws[4,2] = cup/n


deltaU = 0.8*(1-(Ws@np.diag(Ns)).sum(axis=1).max())

M1 = np.linalg.inv(np.eye(n)-np.ones((n,n))*c1*ac/n)
ra1 = (M1@(deltaU*np.ones((n,))/n))[0]

suptitle = ''.join([n+'='+str(round(x,2))+',' for n,x in zip(['c1','c2','rate1','rate2','c3','cup','T','n','c'],
                                                    [c1,c2,rate1,rate2,c3,cup,T,n,c])])+'m='+method


flick_dur = 10e-3
dt = flick_dur/10000

ext_max = 2*(rate1/ra1)*((n+n2)/n)*(1+c)+(rate1/ra1)

############# fine tuning of rates without flicker #########################
p = np.array([1,n2/n,1,n2/n,1,0])*rate1/ra1
p[5] = ext_max-np.sum(p)
p = p / np.sum(p)
T_test = 5
ns = int(T_test/dt)
s,_,_,_ = simulate_model_subnets_nonorm_thresholds(Ns,Ws,p,deltaU,num_steps=ns)
s = np.array(s)
real_rate1 = np.sum(s[:,0])/(n*T_test)
real_rate2 = np.sum(s[:,2])/(n*T_test)
# change dt so that the rates are right
#dt = dt*(real_rate1/rate1)
deltaU = deltaU*(rate1/real_rate1)
print('new deltaU = '+str(deltaU)+' '+str(deltaU + Ws.sum(axis=1).max()))
assert(deltaU+Ws.sum(axis=1).max()<1)



########### now sample with flicker for specified time
#T = 100
flicksa = []
flicksb = []

u = np.random.random((N,))
avss = []
avts = []
kss = []


for it in (range(int(T/flick_dur))):
    #print(it,flush=True)
    if it % int((int(T/flick_dur))/10) == 0:
        print(it,flush=True)

    flicka = (rate1/ra1)*(1+c*flicker())
    flickb = (rate1/ra1)*(1+c*flicker())
    if method=='ext':
        flickb = flickb*1.3
    flicksa.extend([flicka]*int(flick_dur/dt))
    flicksb.extend([flickb]*int(flick_dur/dt))
    p1 = flicka
    p2 = flicka*(n2/n)
    p3 = flickb
    p4 = flickb*(n2/n)
    p5 = 0#2*rate1/ra1
    p = np.array([p1,p2,p3,p4,p5,0])
    p[5] = ext_max - np.sum(p)
    p = p/np.sum(p)
    ns = int(flick_dur/dt)
    avs,avt,u,ks = simulate_model_subnets_nonorm_thresholds(Ns,Ws,p,deltaU,thresh=thresh,num_steps=ns,u0=u)
    avs = np.array(avs)
    kss.append(ks)
    avss.append(avs)
    avts.append(np.array(avt)+it*int(flick_dur/dt))

flicksa = np.array(flicksa)
flicksb = np.array(flicksb)
avs = np.concatenate(avss)
avt = np.concatenate(avts)
ks = np.concatenate(kss)
ns = len(ks)
print(len(avs),flush=True)

spv = np.zeros((5,ns),dtype=int)
finpA = np.zeros((ns,),dtype=int)
finpB = np.zeros_like(finpA)

finpA[ks==0] = 1
finpB[ks==1] = 1

for at,a in zip(avt,avs):
    spv[0,at] = a[0]
    spv[1,at] = a[2]
    spv[2,at] = a[4]

########## Parameters for leaky iaf ##################    
Vthr = 1
tau = 3e-3
dV = 2/(n)
#ext_inputs = np.sum(spv[:2,:],axis=0)
#ext = np.random.random((ns,)) < 100*T*n/ns
#ext = np.random.random((ns,)) < 800*T/ns
ext_inputs = np.sum(spv[:2,:],axis=0)
V,s = sim_iaf(ns,dt,ext_inputs,tau,Vthr=Vthr,dV=dV)
spv[3,s] = 1


print('pop_rates')
print(np.sum(spv[0,:])/(T*n))
print(np.sum(spv[1,:])/(T*n))
print(np.sum(spv[2,:])/(T*n))
print('rate leaky iaf')
print(np.sum(spv[3,:])/(T))

print('subsample to 1ms')
print('this corresponds to '+str(int(1e-3/dt))+' dts')

Ts = len(ks)*dt
bins = int(Ts/1e-3)

flick_dur = int(1e-3/dt)*dt
Bend = int(len(ks)/(flick_dur/dt))#int(T/flick_dur)

fA = bin_data(flicksa,bins)
fB = bin_data(flicksb,bins)
binnedV4_leak = bin_data(spv[3,:],bins)
dt = Ts/bins

pop_act_V4_leak = spv[3,:]

print('parameter values',flush=True)
print(suptitle,flush=True)

print('eq rates',flush=True)
print(pop_act_V4_leak.sum()/T,flush=True)


print('corr coef real flicker input A,V4 B,V4 leak')
ccfAV4_leak =np.corrcoef(fA,binnedV4_leak)[0,1]
ccfBV4_leak = np.corrcoef(fB,binnedV4_leak)[0,1]
print(ccfAV4_leak,flush=True)
print(ccfBV4_leak,flush=True)

tau_k = 15e-3
t_k = np.arange(0,4*tau_k,dt)
Kexp = 1/tau_k*np.exp(-t_k/tau_k);
lfp = [None,None,None,None]

freqs= np.logspace(np.log10(5),np.log10(200),100)
num_pops = 3
lfp[3] = np.convolve(np.array(binnedV4_leak),Kexp,mode='same')


subsample_fact = 1

cwt_V4_leak,_ = wavelet_morlet(lfp[3][::subsample_fact],subsample_fact*dt,freqs)

cwt_fA,_ = wavelet_morlet(np.array(fA[::subsample_fact]),subsample_fact*dt,freqs)
cwt_fB,coi = wavelet_morlet(np.array(fB[::subsample_fact]),subsample_fact*dt,freqs)

c4fA_leak,taus = spectral_coherence(np.array([cwt_fA]),np.array([cwt_V4_leak]),tau_max = 0.2,ntau=4,dt=subsample_fact*dt)
c4fB_leak,taus = spectral_coherence(np.array([cwt_fB]),np.array([cwt_V4_leak]),tau_max = 0.2,ntau=4,dt=subsample_fact*dt)

print('sum of sc values flicker a,V4 vs flicker b,V4 (10ms)') 
print(np.sum(c4fA_leak),flush=True)
print(np.sum(c4fB_leak),flush=True)



res = {}
res['spectral coherences leak'] = [np.sum(np.abs(x)) for x in [c4fA_leak,c4fB_leak]]
res['spectral coherences leak tau=0'] = [x[:,4] for x in [c4fA_leak,c4fB_leak]]

res['correlations leak'] = [ccfAV4_leak,ccfBV4_leak]
res['param_names'] = ['c1','c2','rate1','rate2','c3','cup','T','n','dtcorr','c','method']
res['param_values'] = [c1,c2,rate1,rate2,c3,cup,T,n,dtcorr,c,method]


