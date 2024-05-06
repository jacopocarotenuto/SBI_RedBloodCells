from InternalLibrary.SimulatorPackage import SimulationPipeline

simulator_args = {
    "dt": 1e-6, "DeltaT": 1/25_000, "TotalT": 13, "transient_time": 3
}
# TotalT = 2s, transient_time = 1s

prior_limits = {
    "mu_x": [1.5e4, 4e4],
    "mu_y": [1e4, 140e4],
    "k_x": [3e-3, 16e-3],
    "k_y": [1.5e-2, 30e-2],
    "k_int": [1e-3, 6e-3],
    "tau": [2e-2, 20e-2],
    "eps": [0.5, 6],
    "D_x": [5.5, 15.5],
    "D_y": [1, 530] 
}

simulation_pipeline = SimulationPipeline(batch_size = 200, total_sim = 5000, simulator_args = simulator_args, prior_limits = prior_limits)
simulation_pipeline.start_pipeline_parallel()