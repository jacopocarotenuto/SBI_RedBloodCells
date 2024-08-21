from numba import jit
from numpy import zeros, arange, float32, sqrt, int64
import numpy as np
import time
import os
import _pickle as pickle
import pathos.multiprocessing as multiprocessing
from pathos.pools import ProcessPool
import pathos.profile as pr

# IDEAS: Add metrics to simulation pipeline like: average time per batch, total time, etc...

## REWORK IN PROGRESS
## IMPLEMENTING HEUN METHOD

@jit(nopython = True)
def Simulator_noGPU(dt, DeltaT, TotalT, theta, transient_time = 0,  i_state = None, mu_x = 2.8e4, k_x = 6e-3, kbT = 3.8, debug = False):
    
    
    time_steps_amount           = int64(TotalT/dt)                          # Number of steps
    sampled_point_amount        = int64((TotalT - transient_time)/DeltaT)   # Number of sampled points
    sampling_delta_time_steps   = int64(DeltaT/dt)                          # Number of steps between samples
    transient_time_steps        = int64(transient_time/dt)
    n_sim                       = theta.shape[1]

    if debug:
        print("Time steps amount: ",            time_steps_amount)
        print("Sampled point amount: ",         sampled_point_amount)
        print("Sampling delta time steps: ",    sampling_delta_time_steps)
        print("Transient time steps: ",         transient_time_steps)
        print("Number of simulations: ",        n_sim)
        print("You are currently using ",       theta.shape[1], " free parameters")
    
    
    mu_y    = theta[0]
    k_y     = theta[1]
    k_int   = theta[2]
    tau     = theta[3]
    eps     = theta[4]
    D_x     = kbT * mu_x
    D_y     = kbT * mu_y
    
    
    if theta.shape[1] != n_sim:
        raise Exception("Something went wrong with the detection of the number of simulations")
    
    if transient_time > TotalT:
        raise Exception("Transient time is greater than Total Time")
    
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
    f_trace = zeros((n_sim, sampled_point_amount), dtype = float32)
    y_trace = zeros((n_sim, sampled_point_amount), dtype = float32)

    sampling_counter = int64(1)
    
    # POSSIBLE OPTIM: You could overwrite the last used row of the trace to save memory and not create a proxy array
    
    # TRADEOFF: Memory vs Speed. We can generate the numbers here, or before. Maybe the time is the same... Should test
    
    # CHECK: Benchmark the version with the explicit dx, dy, df and the one with the x, y, f arrays with the calculation in the assigment
    
    # Initialize intermediate step
    
    virtual_x = zeros((n_sim,1), dtype = float32)
    virtual_y = zeros((n_sim,1), dtype = float32)
    virtual_f = zeros((n_sim,1), dtype = float32)
    
    deterministic_dx = zeros((n_sim,1), dtype = float32)
    deterministic_dy = zeros((n_sim,1), dtype = float32)
    deterministic_df = zeros((n_sim,1), dtype = float32)
    
    stochastic_dx = zeros((n_sim,1), dtype = float32)
    stochastic_dy = zeros((n_sim,1), dtype = float32)
    stochastic_df = zeros((n_sim,1), dtype = float32)
    
    virtual_deterministic_dx = zeros((n_sim,1), dtype = float32)
    virtual_deterministic_dy = zeros((n_sim,1), dtype = float32)
    virtual_deterministic_df = zeros((n_sim,1), dtype = float32)
    
    
    # Pre-compute constant
    sqrt_2_D_x_dt = sqrt(2*D_x*dt)
    sqrt_2_D_y_dt = sqrt(2*D_y*dt)
    sqrt_2eps2_dt_tau = sqrt(2*eps**2*dt/tau)
    
    mu_x_dt = mu_x*dt
    mu_y_dt = mu_y*dt
    tau_dt  = dt/tau
    
    
    for t in arange(time_steps_amount - 1):
        # Virtual time-step
        
        deterministic_dx[:,] = ( -k_x * x[:,] + k_int * y[:,] ) * mu_x_dt
        deterministic_dy[:,] = ( -k_y * y[:,] + k_int * x[:,] + f[:,] ) * mu_y_dt
        deterministic_df[:,] = -f[:,]*tau_dt
         
        stochastic_dx[:,] = sqrt_2_D_x_dt * np.random.randn(n_sim,1)
        stochastic_dy[:,] = sqrt_2_D_y_dt * np.random.randn(n_sim,1)
        stochastic_df[:,] = sqrt_2eps2_dt_tau * np.random.randn(n_sim,1)

        virtual_x[:,] =  x[:,] + deterministic_dx[:,] +  stochastic_dx[:,]
        virtual_y[:,] =  y[:,] + deterministic_dy[:,] +  stochastic_dy[:,]
        virtual_f[:,] =  f[:,] + deterministic_df[:,] +  stochastic_df[:,]
        
        virtual_deterministic_dx = ( -k_x * virtual_x[:,] + k_int * virtual_y[:,] ) * mu_x_dt
        virtual_deterministic_dy = ( -k_y * virtual_y[:,] + k_int * virtual_x[:,] + virtual_f[:,] ) * mu_y_dt
        virtual_deterministic_df = -virtual_f[:,]*tau_dt
        
        # Real time-step  x + (A + B)/2 = xt -A/2 + B/2 = xt + (B-A)/2
        x[:,] = virtual_x[:,] + ( virtual_deterministic_dx[:,] - deterministic_dx[:,]) / 2
        y[:,] = virtual_y[:,] + ( virtual_deterministic_dy[:,] - deterministic_dy[:,]) / 2
        f[:,] = virtual_f[:,] + ( virtual_deterministic_df[:,] - deterministic_df[:,]) / 2
        
        # Old Euler Code, keep for reference
        # x[:,] = x[:,] + mu_x*(- k_x * x[:,] + k_int*y[:,])*dt      + sqrt(2*D_x*dt)   * np.random.randn(n_sim,1)
        # y[:,] = y[:,] + mu_y*(-k_y*y[:,] + k_int*x[:,] + f[:,])*dt + sqrt(2*D_y*dt)   * np.random.randn(n_sim,1)
        # f[:,] = f[:,] + -(f[:,]/tau)*dt                            + sqrt(2*eps**2*dt/tau) * np.random.randn(n_sim,1)

        sampling_counter = sampling_counter + 1
        if sampling_counter == sampling_delta_time_steps:
            sampling_counter = int64(1)
            if t >= transient_time_steps:
                x_trace[:, int((t - transient_time_steps)/sampling_delta_time_steps)] = x[:,0]
                f_trace[:, int((t - transient_time_steps)/sampling_delta_time_steps)] = f[:,0]
                y_trace[:, int((t - transient_time_steps)/sampling_delta_time_steps)] = y[:,0]

            

    return x_trace, y_trace, f_trace # Check if this is right


def CheckParameters(dt, DeltaT, TotalT, theta):
    '''
    Checks the variables and parameters for the simulation.

    INPUT
    dt: integration time
    DeltaT: sampling time
    TotalT: total simulation time
    theta: parameters
    '''
    time_steps_amount = int64(TotalT/dt) # Number of steps
    sampled_point_amount = int64(TotalT/DeltaT) # Number of sampled points
    sampling_delta_time_steps = int64(DeltaT/dt) # Number of steps between samples
    n_sim = theta[0].shape[0]
    # Aggiugnere controllo sul TotalT effettivo a fine simulazione
    # Aggiungere controllo sul sampling_delta_time_steps per sanity check
    # Controllare che sampled_point_amount*sampling_delta_time_steps = time_steps_amount
    
    print(f"Your Integration Time (dt) is {dt:.2E} seconds")
    print(f"Your Sampling Time (Delta) is {DeltaT:.2E} seconds, corresponding to a {1/DeltaT:.2f}Hz sampling frequency")
    print(f"Your Total Simulation Time is {TotalT:.2E} seconds")
    print(f"Your Number of Simulated Trajectories is {n_sim:.2E}")
    print(f"The amount of total time steps is {time_steps_amount:.2E}")
    print(f"The amount of sampled points is {sampled_point_amount:.2E}")
    print(f"The gap between two sampled points is {sampling_delta_time_steps:.1E} time steps")
    passed_sanity_checks = True
    print("---- SANITY CHECKS ----")
    if TotalT != DeltaT*sampled_point_amount:
        print(f"WARNING: TotalT is {TotalT}s, but DeltaT*sampled_point_amount is {DeltaT*sampled_point_amount}s")
        passed_sanity_checks = False
    if sampled_point_amount*sampling_delta_time_steps != time_steps_amount:
        print(f"WARNING: sampled_point_amount*sampling_delta_time_steps is {sampled_point_amount*sampling_delta_time_steps}, but time_steps_amount is {time_steps_amount}")
        passed_sanity_checks = False
    if time_steps_amount*dt != TotalT:
        print(f"WARNING: time_steps_amount*dt is {time_steps_amount*dt}, but TotalT is {TotalT}")
        passed_sanity_checks = False
    if dt*sampling_delta_time_steps != DeltaT:
        print(f"WARNING: dt*sampling_delta_time_steps is {dt*sampling_delta_time_steps}, but DeltaT is {DeltaT}")
        passed_sanity_checks = False
    if passed_sanity_checks:
        print("All checks passed")
        
    return None

class SimulationPipeline():
    def __init__(self, batch_size: int, total_sim: int, simulator_args: dict, prior_limits: dict, check_parameters: bool = False):
        self.batch_size = batch_size
        self.total_sim = total_sim
        if batch_size > total_sim:
            print("Your batch size is greater than the total number of simulations, setting batch size to total number of simulations")
            self.batch_size = total_sim
        self.simulator_args = simulator_args
        self.prior_limits = prior_limits
        self.total_batches = total_sim // batch_size
        if total_sim != self.total_batches * batch_size:
            print(f"Actually computing {self.total_batches * batch_size} simulations instead of {total_sim} simulations")
        self.generator = self.create_generator()
        if check_parameters:
            CheckParameters(**simulator_args, theta = self._get_new_theta_batch())
        _, _, _ = Simulator_noGPU(dt = 1, DeltaT = 1, TotalT = 1,theta = self._get_new_theta_batch())
        
    def _get_new_theta_batch(self):
        return np.array([np.random.uniform(self.prior_limits[i][0], self.prior_limits[i][1], size=(self.batch_size, 1)) for i in self.prior_limits])
    
    def _simulate_batch(self, theta):
        x_trace, y_trace, f_trace = Simulator_noGPU(theta = theta, **self.simulator_args)
        return x_trace, y_trace, f_trace

    def create_generator(self):
        def generator():
            for i in range(self.total_batches):
                theta = self._get_new_theta_batch()
                start = time.time()
                x_trace, y_trace, f_trace = self._simulate_batch(theta)
                end = time.time()
                print(f"Simulated batch {i + 1} of {self.total_batches} at {time.strftime('%H:%M:%S of %Y-%m-%d')} in {end - start:.2f} seconds\r",end="")
                yield {"theta": theta, "x_trace": x_trace, "y_trace": y_trace, "f_trace": f_trace, "n_sim": self.batch_size, "time_of_creation": time.strftime("%Y%m%d-%H%M%S")}
        
        return generator
    
    def _save_batch(self, sim):
        simulation_folder = "../../Data/Simulations"
        today_folder = time.strftime("%Y%m%d")

        if not os.path.exists(simulation_folder):
            os.makedirs(simulation_folder)

        today_folder_path = os.path.join(simulation_folder, today_folder)
        if not os.path.exists(today_folder_path):
            os.makedirs(today_folder_path)
        
        
        file_name = os.path.join(today_folder_path, f"{sim['time_of_creation']}_{sim['n_sim']}sims_process"+str(pr.process_id())+".pkl")    
        # Save the batch
        if os.path.exists(file_name):
            file_name = os.path.join(today_folder_path, f"{sim['time_of_creation']}_{sim['n_sim']}sims_"+str(round(time.time()))+".pkl")
        
        with open(file_name, "wb") as f:
            pickle.dump(sim, f, protocol=2)
        #print(f"Saved batch to {file_name}")
        return None
    
    def start_pipeline(self):
        print(f"Starting simulation pipeline at {time.strftime('%Y%m%d-%H%M%S')} with {self.total_sim} simulations in batches of size {self.batch_size}")
        start = time.time()
        for sim in self.generator():
            self._save_batch(sim)
        end = time.time()
        print(f"Finished simulating {self.total_sim} simulations at {time.strftime('%H:%M:%S of %Y-%m-%d')} in {end - start:.2f} seconds")
    
    def __str__(self):
        return f"SimulationPipeline with {self.total_sim} simulations in batches of size {self.batch_size}"
    
    def start_pipeline_parallel(self, cores = -1):
        print(f"Starting simulation pipeline at {time.strftime('%Y%m%d-%H%M%S')} with {self.total_sim} simulations in batches of size {self.batch_size}")
        
        if cores == -1:
            cores = multiprocessing.cpu_count()
        if cores > multiprocessing.cpu_count():
            print(f"WARNING: You are using {cores} cores, but you have only {multiprocessing.cpu_count()} cores available")
            
        if self.total_batches % cores != 0:
            print("WARNING: The number of batches is not divisible by the number of cores, this leads to sub-optimal performance")
        pool = ProcessPool(nodes=cores)
        
        start = time.time()
        print(f"Simulating {self.total_batches} batches in parallel...")
        with ProcessPool(nodes=cores) as pool:
            pool.map(self._simulate_and_save_batch, [self._get_new_theta_batch()]*self.total_batches)
        end = time.time()
        print(f"Finished simulating {self.total_sim} simulations at {time.strftime('%H:%M:%S of %Y-%m-%d')} in {end - start:.2f} seconds")
    
    def _simulate_and_save_batch(self, theta):
        x_trace, y_trace, f_trace = self._simulate_batch(theta)
        self._save_batch({"theta": theta, "x_trace": x_trace, "y_trace": y_trace, "f_trace": f_trace, "n_sim": self.batch_size, "time_of_creation": time.strftime("%Y%m%d-%H%M%S")})
        print(f"Saved batch of {self.batch_size} simulations at {time.strftime('%H:%M:%S of %Y-%m-%d')}")
        
class SimulationLoader():
    def __init__(self, day_folder = None) -> None:
        self.simulation_folder = "Simulation"
        if day_folder is None:
            self.today_folder = time.strftime("%Y%m%d")
        else:
            self.today_folder = day_folder
        self.today_folder = time.strftime("%Y%m%d")
        self.today_folder_path = os.path.join(self.simulation_folder, self.today_folder)
        self.simulation_files = [os.path.join(self.today_folder_path, i) for i in os.listdir(self.today_folder_path)]
    
    def load_all_simulation(self):
        for file in self.simulation_files:
            with open(file, "rb") as f:
                sim = pickle.load(f)
                print("Loaded simulation from ", file)
            yield sim
    
    def load_n_simulations(self, n):
        for file in self.simulation_files[:n]:
            with open(file, "rb") as f:
                sim = pickle.load(f)
                print("Loaded simulation from ", file)
            yield sim
    
    def load_n_random_simulations(self,n):
        for file in np.random.choice(self.simulation_files, n):
            with open(file, "rb") as f:
                sim = pickle.load(f)
                print("Loaded simulation from ", file)
            yield sim