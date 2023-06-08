import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

def wavelet_morlet(sig,dt,freqs,w=6.0,return_coi=True):
    """sig is 1d array, dt step size in seconds, freqs in Hz
       returns complex wavelet coefficients."""
    sig = (sig-sig.mean())/sig.std()
    fs = 1/dt
    widths = w*fs/(2*np.pi*freqs)
    a = signal.cwt(sig,signal.morlet2,widths,w=w)
    if return_coi:
        coi = np.sqrt(2)*widths*dt
        return a,coi
    return a


def plot_wavelet_with_coi(a,coi,t,freqs,ax=None,**pcm_kwargs):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    im = ax.pcolormesh(t, freqs, np.abs(a), cmap='viridis', shading='gouraud',**pcm_kwargs)
    coil = np.where(coi>t[-1],t[-1],coi)
    coir = t[-1]-coi
    coir = np.where(coir<0,0,coir)
    ax.plot(coil,freqs,'k--')
    ax.plot(coir,freqs,'k--')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('frequency [Hz]')
    return im


def spectral_coherence(a1,a2,tau_max,ntau,dt,ts=None,te=None):
    """a1 and a2 are (trials,freqs,time) arrays, ts and te time start/end indices
       tau_max is the maximal time delay (dimension time)
       returns C of shape (freqs,taus) where taus is of lenght 2*ntau+1 linearly from -tau_max to +tau_max
        """
    ntrails,nfreqs,ntime = a1.shape
    assert(a2.shape==a1.shape)
    if ts is None:
        ts = 0
    if te is None:
        te = ntime
    #tau_max = 0.6
    #ntau = int(tau_max/dt)
    taus = np.linspace(-tau_max,tau_max,2*ntau+1)
    # for single trial
    c = np.zeros((nfreqs,len(taus)))

    #helper for sum of complex dot products over all trials
    def trial_dot_sum(p,q):
        return np.sum([p[i,:]@np.conj(q[i,:]) for i in range(p.shape[0])])

    for i in range(nfreqs):
        # zero time delay at index ntau
        c[i,ntau] = (np.abs(trial_dot_sum(a1[:,i,ts:te],a2[:,i,ts:te]))**2/
                     (np.sum(np.abs(a1[:,i,ts:te])**2)*np.sum(np.abs(a2[:,i,ts:te])**2)))
        for itau in range(1,ntau+1):
            tau = taus[ntau+itau] # absolute tau value
            taui = int(tau/dt) # index shift by tau
            # shift by +tau
            c[i,ntau+itau] = (np.abs(trial_dot_sum(a1[:,i,(ts+taui):te],a2[:,i,ts:(te-taui)]))**2/
                              (np.sum(np.abs(a1[:,i,(ts+taui):te])**2)*np.sum(np.abs(a2[:,i,ts:(te-taui)])**2)))
            # shift by -tau -> switch t boundaries
            c[i,ntau-itau] = (np.abs(trial_dot_sum(a1[:,i,ts:(te-taui)],a2[:,i,(ts+taui):te]))**2/
                              (np.sum(np.abs(a1[:,i,ts:(te-taui)])**2)*np.sum(np.abs(a2[:,i,(ts+taui):te])**2)))            
    # nonlinear normalization    
    #C = 1/(1+np.sqrt(1/c - 1))
    return c,taus
