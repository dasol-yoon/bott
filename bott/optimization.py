#!/usr/bin/env python3

import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import (
    LogExpectedImprovement,  # See https://arxiv.org/abs/2310.20708 for details.
)
from botorch.acquisition import qKnowledgeGradient

# from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.logei import (
    qLogExpectedImprovement,  # See https://arxiv.org/abs/2310.20708 for details.
)
from botorch.acquisition.objective import GenericMCObjective, MCAcquisitionObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import (
    SingleTaskGP,  # No FixedNoiseGP in botorch 0.13.0
)
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from bott.loss import MSE, NRMSE, SSE
from bott.ts_acqf import ThompsonSampling
from bott.utils import time_sync


def run_one_trial(
        problem_name: str,
        problem: SyntheticTestFunction,
        algo: str,
        trial: int,
        n_init_evals: int,
        max_iter: int,
        metrics: List[str]=['obs_val'],
        objective: Optional[MCAcquisitionObjective] = None,
        noisy: Optional[bool]=False,
        device_botorch: str='cpu',
        dtype: torch.dtype = torch.double
)-> None:
    '''Run one trial of BO loop for the given problem (tile pattern) and algorithm

    Args:
        problem_name: tile pattern name
        problem: A problem class that defines which tile pattern to use
        algo: A string representing the name of the algorithm: [EI, KG, Random, EICF, EICFT]
        trial: The seed of the trial
        metrics: A list of metrics to record: ['pos_mean', 'obs_val']
        n_init_evals: The number of initial evaluations
        max_iter: The number of maximum iterations for optimization loop
        noisy: A boolean indicating whether the objective function evaluation are noisy

    Returns:
        None.
    '''

    time_init_start = time_sync()
    
    # Setup
    device = torch.device(device_botorch)
    current_directory = os.getcwd()
    results_dir = f"{current_directory}/results/{problem_name}/{algo}/"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Acquisition algo: {algo}")
    print(f"Trial seed: {trial}")
    print(f"problem.device : {problem.device}")
    print(f"GP model device: {device_botorch}")
    
    
    if objective is None:
        objective = GenericMCObjective(lambda Y, X=None: ((Y[...,:-1]-problem.reduction_true.to(device)).pow(2)+Y[...,[-1]]).sum(dim=-1))

    # set random seed
    torch.manual_seed(trial)
    np.random.seed(trial)
    random.seed(trial)

    # random initial points and calculate intermediate outputs [cbed]
    X=draw_sobol_samples(bounds=problem.bounds.to(device=device),n=n_init_evals,q=1).squeeze(-2) # X = [n_init, problem.dim], note that X is by default on cpu because problem.bounds is also on cpu

    # Physical images output 
    input_params = X.tolist()
    outputs_np = np.array([problem.get_physics_simu(*param) for param in input_params])
    image_output = torch.tensor(outputs_np, dtype=dtype, device=device)
    if noisy:
        image_output = image_output + torch.normal(0,1,size=image_output.shape)
    
    # calculate final objective (SSE), this is pixelSSE. Might consider rename SSE_value into pixel_losses, and make reduction_SSE into group_loss.
    SSE_value  = SSE(y_simu=image_output, y_true=problem.measurement_true.to(device), dp_pow=1, reduce=False).unsqueeze(-1) # SSE_value = [n_init, 1]

    # calculate reduction intermediate outputs for EICF in batch
    if algo=='EICF':
        y_reduction = problem.reduction_func(image_output).to(device)  # [n_init, num_tiles]
        reduction_SSE = SSE(y_simu=y_reduction, y_true=problem.reduction_true.to(device), dp_pow=1, reduce=False).unsqueeze(-1) # [n_init, 1]
        epsilon = SSE_value - reduction_SSE
        y_value = torch.cat((y_reduction,epsilon),dim=-1) # [n_init, num_tiles+1]
            
    obj = -1*SSE_value # maximization direction
    best_vals = [obj.max().item()] # list of values
    best_val = obj.max() # tensor
    acqf_vals = []
    acqf_runtime = []
    
    time_init_end = time_sync()
    print(f"Initializing model with '{n_init_evals}' initial evaluations took {time_init_end - time_init_start:.3f} sec")
    
    # Start the optimization loop
    for iter in range(max_iter):
        
        time_iter_start = time_sync()
        
        # Choose the acquisition function algorithm
        if algo in ['EI','KG','Random','TS']:
            train_Y = SSE_value
        else:
            train_Y = y_value
        
        # Get model #TODO Do we really need to create model with every iteration?
        time_model_start = time_sync()
        model = SingleTaskGP(train_X=X, train_Y=train_Y, train_Yvar=torch.ones_like(train_Y) * 0.0001, # #TODO Need to configure this variance scaling hyperparameter
                        outcome_transform=Standardize(m=train_Y.shape[-1]),input_transform=Normalize(d=X.shape[-1])).to(device) # GP is on y, not the objective
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        time_model_end = time_sync()
        
        # Get new sample with timing
        time_sample_start = time_sync()
        new_x, acqf_val = get_new_sample(model=model,algo=algo,problem=problem,best_val=best_val,objective=objective, device=device, dtype=dtype)
        time_sample_end = time_sync()
        
        # Append and concat new values       
        acqf_vals.append(acqf_val)
        acqf_runtime.append(time_sample_end - time_sample_start)
        X = torch.cat((X, new_x),dim=0)
        
        # Run physical model with a new_x
        input_param = new_x.tolist()[0] # [value0, value1, value2]
        time_simu_start = time_sync()
        image_temp = torch.from_numpy(problem.get_physics_simu(*input_param)).to(dtype=dtype, device=device)
        time_simu_end = time_sync()
        # image_output = torch.cat((image_output, image_temp.unsqueeze(0)),dim=0) # This will continue to concat new images but image_output is never used. We should remove this unless it's needed somewhere else.
        
        # calculate final objective (SSE)
        new_SSE = SSE(y_simu=image_temp,y_true=problem.measurement_true.to(device),dp_pow=1).unsqueeze(0).unsqueeze(0) # [1,1]
        SSE_value = torch.cat((SSE_value, new_SSE), dim=0)

        # calculate reduction intermediate outputs for EICF
        if algo=='EICF':
            y_reduction = problem.reduction_func(image_temp).to(device) # [num_tiles,]
            reduction_SSE = SSE(y_simu=y_reduction,y_true=problem.reduction_true.to(device),dp_pow=1).unsqueeze(0) # [1,]
            epsilon = SSE_value[-1] - reduction_SSE
            y_temp = torch.cat((y_reduction,epsilon),dim=-1).unsqueeze(0) # [1,num_tiles+1]
            y_value = torch.cat((y_value,y_temp),dim=0)
        
        # Display and save results
        obj = -1*SSE_value
        best_vals.append(obj.max().item())
        best_val = obj.max()
        
        time_iter_end = time_sync()
        
        print(f"\nIteration: {iter+1}/{max_iter}")
        print(f"GP model fitting time: {time_model_end - time_model_start:.3f} sec")
        print(f"Acqu func sampling time: {time_sample_end - time_sample_start:.3f} sec")
        print(f"Physics simulation time: {time_simu_end - time_simu_start:.3f} sec")
        print(f"Iteration time: {time_iter_end - time_iter_start:.3f} sec")
        print(f"Suggested point: {new_x}")
        print(f"new SSE value: {SSE_value[-1]}")
        if algo not in ['EI','KG','Random','TS']:
            print(f"Reduction valued: {y_temp}")
        print(f"Best objective function value found: {-1*best_val}")
        print(f"==========================================================")
        BO_results = {
            "max_iter": max_iter,
            "acqf_runtime": acqf_runtime,
            "acqf_val_list": acqf_vals,
            "best_obs_vals": best_vals,
            "train_X": X,
            "train_Y": train_Y,
            "obj_func_val": obj,
            "random_states": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
        }
        torch.save(BO_results, results_dir + f"trial_{trial}.pt")
    print(f"\nTotal run time with '{max_iter}' iters: {time_sync() - time_init_start:.3f} sec")

def get_new_sample(model,algo, problem,best_val,objective, device='cpu', dtype=torch.double):
    '''Produce a new sample or batch to evaluate the objective function

    Args:
        - model: A GP model
        - algo: The name of algorithm
        - problem: A problem class that defines which tile pattern to use
    Returns:
        - a tuple consisting of a tensor of the new suggested input to evaluate or batch of inputs and the corresponding acquisition value
    '''
    
    if algo == 'EI':
        acqf = LogExpectedImprovement(model=model,best_f=best_val)
        new_x, acqf_val = optimize_acqf(acq_function=acqf,bounds=problem.bounds.to(dtype=dtype, device=device),q=1,num_restarts=20,raw_samples=100)        
        return new_x, acqf_val
    elif algo == 'KG':  
        acqf = qKnowledgeGradient(model, num_fantasies=128)
        new_x, acqf_val = optimize_acqf(acq_function=acqf,bounds=problem.bounds.to(dtype=dtype, device=device),q=1,num_restarts=10,raw_samples=512)
        return new_x, acqf_val 
    elif algo == 'EICF':
        sampler = SobolQMCNormalSampler(torch.Size([1024])).to(dtype=dtype, device=device)
        EICF = qLogExpectedImprovement(model=model, best_f=best_val, objective=objective,sampler=sampler).to(dtype=dtype, device=device)
        new_x, acqf_val = optimize_acqf(
            acq_function=EICF,
            bounds=problem.bounds.to(dtype=dtype, device=device),
            q=1,
            num_restarts=50,
            raw_samples=100,
        )
        return new_x, acqf_val
    elif algo =='TS':
        acq_function = ThompsonSampling(model=model)
        new_x, acqf_val = optimize_acqf(
            acq_function=acq_function,
            bounds=problem.bounds.to(dtype=dtype, device=device),
            q=1,
            num_restarts=20,
            raw_samples=100,
            options={"batch_limit": 1},
        )
        return new_x, acqf_val 
    elif algo == 'Random':
        new_x = (
            torch.rand([1, problem.dim]) * (problem.bounds[1] - problem.bounds[0])
            + problem.bounds[0]
        ).to(dtype=dtype, device=device)
        return new_x, None
    else:
        raise ValueError(f"The current implementation does not support algo = '{algo}', please use either 'EI, 'EICF', or 'Random'")