#!/usr/bin/env python3

import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective, MCAcquisitionObjective
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler  # import package
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
print(device)

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

    current_directory = os.getcwd()
    results_dir = f"{current_directory}/results/{problem_name}/{algo}/"
    os.makedirs(results_dir, exist_ok=True)

    if objective is None:
        objective = GenericMCObjective(lambda Y: Y[...,-1])

    # set random seed
    torch.manual_seed(trial)
    np.random.seed(trial)
    random.seed(trial)

    # random initial points and calculate intermediate outputs
    X=draw_sobol_samples(bounds=torch.Tensor(problem.bounds),n=n_init_evals,q=1).squeeze(-2) 
    inter_output = problem.evaluate(X) # tensor

    input_dim = X.shape[-1]
    output_dim = inter_output.shape[-1]

    if noisy:
        inter_output = inter_output + torch.normal(0,1,size=inter_output.shape)
    
    # compute objective function values and extract the current best maximum value
    obj = objective(inter_output)
    best_vals = [obj.max().detach().item()]
    best_val = obj.max()
    acqf_vals = []
    acqf_runtime = []
    for iter in range(max_iter):
        model = FixedNoiseGP(X, inter_output, torch.ones_like(inter_output) * 0.0001,
                             outcome_transform=Standardize(m=output_dim),input_transform=Normalize(d=input_dim)) # GP is on y, not the objective
        fit_gpytorch_model(ExactMarginalLogLikelihood(model.likelihood, model))
        start_time = time.time()
        new_x, acqf_val = get_new_sample(model=model,algo=algo,problem=problem,best_val=best_val,objective=objective)
        running_time = time.time()-start_time
        acqf_vals = acqf_vals+[acqf_val]
        acqf_runtime = acqf_runtime + [running_time]
        X = torch.cat((X,new_x),dim=0)
        new_y = problem.evaluate(new_x)
        inter_output = torch.cat((inter_output,new_y),dim=0)
        obj = objective(inter_output)
        best_vals = best_vals+[obj.max().detach().item()]
        best_val = obj.max()
        print(f"Iteration: {iter}/{max_iter}")
        print(f"Suggested point: {new_x}")
        print(f"Intermediate output: {new_y}")
        print(f"Objective function value: {objective(new_y)}")
        print(f"Best objective function found: {best_val}")
        print(f"==========================================================")
        BO_results = {
            "max_iter": max_iter,
            "acqf_runtime": acqf_runtime,
            "acqf_val_list": acqf_vals,
            "best_obs_vals": best_vals,
            "train_X": X,
            "train_Y": inter_output,
            "obj_func_val": obj,
            "random_states": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
        }
        torch.save(BO_results, results_dir + f"trial_{trial}.pt")

def get_new_sample(model,algo, problem,best_val,objective):
    '''Produce a new sample or batch to evaluate the objective function

    Args:
        - model: A GP model
        - algo: The name of algorithm
        - problem: A problem class that defines which tile pattern to use
    Returns:
        - a tuple consisting of a tensor of the new suggested input to evaluate or batch of inputs and the corresponding acquisition value
    '''
    if algo == 'EI':
        acqf = ExpectedImprovement(model=model,best_f=best_val)
        new_x, acqf_val = optimize_acqf(acq_function=acqf,bounds=problem.bounds)
        return new_x, acqf_val
    elif algo == 'EICF' or 'EICFT':
        sampler = SobolQMCNormalSampler(1024)
        EICF = qExpectedImprovement(model=model, best_f=best_val, objective=objective,sampler=sampler)
        new_x, acqf_val = optimize_acqf(
            acq_function=EICF,
            bounds=problem.bounds,
            q=1,
            num_restarts=50,
            raw_samples=100,
        )
        return new_x, acqf_val
    elif algo == 'Random':
        new_x = (
            torch.rand([1, problem.input_dim]) * (problem.bounds[1] - problem.bounds[0])
            + problem.bounds[0]
        )
        return new_x, None






'''
What should problem class have?
- attributes: input dimension, output dimension
- functions: that outputs intermediate outputs, there is no need for a wrap up objective
                it should also have for loop if multiple inputs are supplied.
'''