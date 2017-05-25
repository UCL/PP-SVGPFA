import numpy as np
""" Functions to simulate point processes """


def simulate_from_log_rate(log_rate,T,rate_max=None, link=np.exp):
    """ Sampling of a 1d point process via thinning
    :param log_rate: invlink rate function
    :param T: Time horizon
    :param rate_max: maximum rate if known (for ex, if link is bounded)
    :param link: link function, output is instantaneous rate
    """
    # maximum rate is estimated if not given
    time_grid = np.linspace(0,T,300)
    rate_max = rate_max if rate_max is not None else link(log_rate(time_grid).max())
    # sample homogenous poisson with rate rate_max
    ds = np.random.exponential(1/rate_max, int(2*T*rate_max))
    ts = np.cumsum(ds)
    ts = ts[np.where(ts<T)]
    # rejection sampling
    Ia = np.random.rand(len(ts),)<link(log_rate(ts))/rate_max
    ts = ts[Ia]
    return np.asarray(ts).astype(np.float32).reshape(-1,1)

def log_rates_from_latent(fs,C,d):
    """
    Construct functions gs =  C * fs + d
    :param fs: list of functions of size D
    :param C: matrix of size O x D
    :param d: vector of size O
    :return: list functions of size O
    """
    O,D = C.shape
    log_rates = []
    lr = lambda o: lambda x: np.sum([C[o,d_]*fs[d_](x)+d[o] for d_ in range(D)],0)
    for o in range(O):
        log_rates.append(lr(o))
    return log_rates

def simulate_pp_GPFA_from_latent(fs,C,d,T=1.,R=1,link=np.exp):
    """ Given latent functions fs, loading matrix C and offset d, simulate independent point processes
     with rates C*fs + d
    :param fs: list of functions of size D
    :param C: loading matrix of size O x D
    :param d: offset of size O
    :param T: time horizon
    :param: link: link function giving instantaneous rate
    :return: list of spike times of size D
    """
    log_rates = log_rates_from_latent(fs,C,d)
    TTs = []
    for r in range(R):
        Ts=[]
        for o in range(C.shape[0]):
            Ts.append(simulate_from_log_rate(log_rates[o],T,link=link))
        TTs.append(Ts)
    return TTs, log_rates

def simulate_GPFA_rates_from_latent(fs, C, d, T, R=1, bin_width=0.1 ,link=np.exp):
    """
    Simulate from binned GPFA 
    :param fs: list of functions of size D
    :param C: loading matrix of size O x D
    :param d: offset vector of size O
    :param T: time horizon
    :param R: number of trials with shared log rate
    :param bin_width: width of discretization window
    :param link: link function giving the instantaneous log rate
    :return: 
    """
    O,D = C.shape
    N = int(T/bin_width)
    X_np = np.linspace(0,T,N).reshape(-1, 1)
    F_np = np.hstack([fs[d_](X_np) for d_ in range(D)])
    Pred_np = np.dot(C, F_np.T).T + d.T
    Pred_np = np.tile(np.expand_dims(Pred_np, 2), (1, 1, R))
    Rate_np = link(Pred_np)
    return X_np, Rate_np, Pred_np


def spikes_in_mat(Ts):
    ''' concatenates spikes of all neurons in a trial
    :param Ts: R-list of [ O-list  of spike counts ]
    :return: concatenates spikes of each trial as matrix of size N x R
    N is the largest total number of spikes in a trial
    '''
    n_trials = len(Ts) # number of trials
    n_out = len(Ts[0])
    Ts_cat  = [np.concatenate(t) for t in Ts] # concatenates neurons spikes
    ind_cat = [np.concatenate([np.ones_like(o,dtype=int)*i  for i,o in enumerate(t)]) for t in Ts] # indices of neuron in concatenation
    max_spikes = np.max([len(t) for t in Ts_cat]) # max number of spikes per trial
    ind_mat =np.ones((max_spikes,n_trials))*-1
    Ts_mat = np.zeros((max_spikes,n_trials))
    for r in range(n_trials):
        ind_mat[0:len(ind_cat[r]),r] = ind_cat[r].flatten()
        Ts_mat[0:len(Ts_cat[r]),r] = Ts_cat[r].flatten()
    mask_mat = np.zeros((max_spikes,n_out,n_trials))
    for o in range(n_out):
        for r in range(n_trials):
           mask_mat[np.where(ind_mat[:,r]==o),o,r]=1
    return Ts_mat,mask_mat

def bin_single_neuron(a,tmin,tmax, bin_width=100.):
    '''
    Binning spike times
    :param a: vector of spike times
    :param tmin: start time, scalar
    :param tmax: end time, scalar
    :param bin_width: bin width, scalar
    :return: counts in bins, vector
    '''
    assert tmax>tmin
    n = int((tmax-tmin)/bin_width)
    e = tmin + np.arange(n)*bin_width
    h,e = np.histogram(a,e)
    return h

def bin_single_trial(t,tmin,tmax, bin_width=100.):
    '''
    Binning single trial
    :param t: list of spike times, len O
    :param tmin: start time, scalar
    :param tmax: end time, scalar
    :param bin_width: bin width, scalar
    :return: matrix of counts, size N x 0
    '''
    n = int((tmax-tmin)/bin_width)
    O = len(t)
    B = np.zeros((n-1,O))
    for o in range(O):
        B[:,o] = bin_single_neuron(t[o],tmin,tmax, bin_width)
    return B

def bin_subject(s,tmin,tmax, bin_width=100.):
    """
    Binning subject
    :param s: list of [list of spike counts, length O] , length R 
    :param tmin: start time, scalar
    :param tmax: end time, scalar
    :param bin_width: bin width, scalar
    :return: matrix of counts, size N x 0 x R
    """
    n = int((tmax-tmin)/bin_width)
    R = len(s)
    O = len(s[0])
    B = np.zeros((n-1,O,R))
    for r in range(R):
        B[:,:,r] = bin_single_trial(s[r],tmin,tmax, bin_width)
    return B


if __name__ == "__main__":

    D = 2 # number of additive terms
    R = 1 # number of trials
    O = 30 # numbre of neurons
    C = np.random.randn(O,D)*.5
    d = np.ones((O,1))*0.5
    T = 10
    p=3.
    fs = [lambda x:np.sin(x*2*np.pi/p),
      lambda x:np.sin(x*2*np.pi/p/2),
      lambda x:np.sin(x*2*np.pi/p/4)]


    T_pp, _ = simulate_pp_GPFA_from_latent(fs,C,d,T=1.,link=np.exp)

    T_bp, Y_bp, _ = simulate_GPFA_rates_from_latent(fs,C,d,T=1.,link=np.exp, bin_width=0.1)

    print('neurons:%d,trials:%d,latents:%d'%(O,R,D))
    print('size of sample of binned GPFA:',Y_bp.shape)
