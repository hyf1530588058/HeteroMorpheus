import random
import numpy as np
import datetime

from ga.run_mybody11 import run_mybody

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    run_mybody(
        structure_shape = (5,5),
        experiment_name = "test_amorpheus",
        max_evaluations = 1,
        train_iters = 50,
        num_cores = 1
    )

    print('run_universal over at ', datetime.datetime.now())

# * `seed` = seed to control randomness
# * `structure_shape` = each robot is represented by `(m,n)` matrix of voxels 
# * `experiment_name` = all experiment files are saved to `saved_data/experiment_name`
# * `max_evaluations` = maximum number of unique robots to evaluate. Should be a multiple of `pop_size`
# * `train_iters` = number of iterations of ppo to train each robot's controller
# * `num_cores` = number of robots to train in parallel. Note: the total number of processes created will be `num_cores * num_processes` (as specified below in the command line)