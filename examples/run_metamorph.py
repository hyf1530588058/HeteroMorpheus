import random
import numpy as np

from ga.run_meta import run_meta

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    run_meta(
        structure_shape = (5,5),
        experiment_name = "test_metamorph",
        max_evaluations = 12,
        train_iters = 1000,
        num_cores = 12
    )

# * `seed` = seed to control randomness
# * `structure_shape` = each robot is represented by `(m,n)` matrix of voxels 
# * `experiment_name` = all experiment files are saved to `saved_data/experiment_name`
# * `max_evaluations` = maximum number of unique robots to evaluate. Should be a multiple of `pop_size`
# * `train_iters` = number of iterations of ppo to train each robot's controller
# * `num_cores` = number of robots to train in parallel. Note: the total number of processes created will be `num_cores * num_processes` (as specified below in the command line)