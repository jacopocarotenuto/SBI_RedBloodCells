import multiprocessing
import time
from Utils_functions import Simulator_noGPU
# dt, DeltaT, TotalT, n_sim, theta


def MultSimulator(dt, DeltaT, TotalT, n_sim, theta, i_state = None):
    Simulator_noGPU(1,1,1,1,[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5], None)
    Simulator_noGPU(dt, DeltaT, TotalT, n_sim, theta, i_state)

def PerformanceTest(n):
    start = time.time()
    Simulator_noGPU(1,1,1,1,[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5], None)
    for _ in range(n):
        a = Simulator_noGPU(1e-5,1e-2,1,1,[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5], None)
    end = time.time()
    print(f"Time taken: {end-start}")



    
if __name__ == '__main__':
    
    PerformanceTest(600)
    
   
    
    cores = multiprocessing.cpu_count()
    start = time.time()
    with multiprocessing.Pool(cores) as pool:
        pool.starmap(MultSimulator, [(1e-5,1e-2,1,1,[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5], None)]*600)
    end = time.time()
    print(f"Time taken: {end-start}")