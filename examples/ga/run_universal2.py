import os
import torch
import numpy as np
import shutil
import random
import math
from ppo.evaluate import evaluate
import sys    
curr_dir = os.path.dirname(os.path.abspath(__file__))  
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)   
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
import datetime
from ppo.myPPOrun2 import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
from ppo.myPPOmodel2 import Policy
from ppo.arguments import get_args
from ppo.envs import make_vec_envs
import itertools
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device=torch.device("cpu")
def run_universal2(experiment_name, structure_shape, max_evaluations, train_iters, num_cores):  
    print()

    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)
     
    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    ### STORE META-DATA ##
    if not is_continuing:    
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")  
        
        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))  
        except:
            pass

        f = open(temp_path, "w")    
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:    
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:     
            if count == 0:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 1:
                max_evaluations = int(line.split()[1])
            if count == 2:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max evals: {max_evaluations}, train iters {train_iters}.')
        
        f.close()

    structures = []    
    num_evaluations = 0    
    #generate a population
    if not is_continuing:  
        for i in range(max_evaluations):
            save_path_structure = os.path.join(root_dir,"robot_universal/walker",str(i) + ".npz")
            np_data = np.load(save_path_structure)    
            structure_data = []
            for key, value in itertools.islice(np_data.items(), 2):  
                structure_data.append(value)
            structure_data = tuple(structure_data)            
            structures.append(Structure(*structure_data, i)) 

    args = get_args()    
    actor_critic = Policy(
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    #actor_critic.load_state_dict(torch.load(os.path.join(root_dir,"saved_data","49","controller","robot_"+str(4)+"_controller"+".pt"))[0].state_dict())
    while True:
        
        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "structure")  
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "controller")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        ### SAVE POPULATION DATA ###
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)   

        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()
        for structure in structures:           
            ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label),actor_critic,args)  
            group.add_job(run_ppo, ppo_args, callback=structure.set_reward)   
                        
        group.run_jobs(num_cores)  

        #not parallel
        #for structure in structures:
        #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)   
        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()

        return
    