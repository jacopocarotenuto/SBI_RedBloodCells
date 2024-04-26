# Load the minimum required library to run the functions
from numba import jit
from numpy import zeros, arange, uint8, int32, float32, sqrt, uint32, ones, int64, mean, ceil, where, log2, max, min
from numpy.random import randn
from numpy.fft import fft, ifft

@jit(nopython = True)
def Simulator_noGPU(dt, DeltaT, TotalT, n_sim, theta, i_state = None):
    
    time_steps_amount = int64(TotalT/dt) # Number of steps
    sampled_point_amount = int64(TotalT/DeltaT) # Number of sampled points
    sampling_delta_time_steps = int64(DeltaT/dt) # Number of steps between samples
    
    # Aggiugnere controllo sul TotalT effettivo a fine simulazione
    # Aggiungere controllo sul sampling_delta_time_steps per sanity check
    # Controllare che sampled_point_amount*sampling_delta_time_steps = time_steps_amount
    
        
    # Unpack Parameters
    mu_x = theta[0]
    mu_y = theta[1]
    k_x = theta[2]
    k_y = theta[3]
    k_int = theta[4]
    tau = theta[5]
    eps = theta[6]
    D_x = theta[7]
    D_y = theta[8]
    
    # Handle initial state
    if i_state is None:
        x = zeros((n_sim,1), dtype = float32)
        y = zeros((n_sim,1), dtype = float32)
        f = zeros((n_sim,1), dtype = float32)
    else:
        if x.shape == (n_sim,1) or y.shape == (n_sim,1) or f.shape == (n_sim,1):
            raise Exception("Initial state has wrong shape, each should be (n_sim,1)")
        x, y, f = i_state
    
    # Initialize x_trace array
    
    x_trace = zeros((n_sim, sampled_point_amount), dtype = float32)
    
    sampling_counter = int64(1)
    
    # POSSIBLE OPTIM: You could overwrite the last used row of the trace to save memory and not create a proxy array
    
    # TRADEOFF: Memory vs Speed. We can generate the numbers here, or before. Maybe the time is the same... Should test
    
    # CHECK: Benchmark the version with the explicit dx, dy, df and the one with the x, y, f arrays with the calculation in the assigment
    
    for t in arange(time_steps_amount - 1):
        x[:,] = x[:,] + mu_x*(- k_x * x[:,] + k_int*y[:,])*dt                +          sqrt(2*mu_x*D_x*dt)     *randn(n_sim,1)
        y[:,] = y[:,] + mu_y*(-k_y*y[:,] + k_int*x[:,] + f[:,])*dt        +          sqrt(2*mu_y*D_y*dt)     *randn(n_sim,1)
        f[:,] = f[:,] + -(f[:,]/tau)*dt                                   +          sqrt(2*eps**2*dt/tau)   *randn(n_sim,1)

        sampling_counter = sampling_counter + 1
        if sampling_counter == sampling_delta_time_steps:
            x_trace[:, int(t/sampling_delta_time_steps)] = x[:,0]
            sampling_counter = int64(1)
    
    return x_trace, (x, y, f) # Check if this is right

def corr(x,y,nmax,dt=False):
    '''fft, pad 0s, non partial'''

    assert len(x)==len(y)

    n=len(x)
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**np.ceil(np.log2(ext_size)).astype('int')

    xp=x-np.mean(x)
    yp=y-np.mean(y)

    # do fft and ifft
    cfx=np.fft.fft(xp,fsize)
    cfy=np.fft.fft(yp,fsize)
    if dt != False:
        freq = np.fft.fftfreq(n, d=dt)
        idx = np.where((freq<-1/(2*dt))+(freq>1/(2*dt)))[0]
        
        cfx[idx]=0
        cfy[idx]=0
        
    sf=cfx.conjugate()*cfy
    corr=np.fft.ifft(sf).real
    corr=corr/n

    return corr[:nmax]

def normalize_numpy(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))