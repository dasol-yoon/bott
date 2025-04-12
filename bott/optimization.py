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
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler  # import package
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from bott.loss import MSE, NRMSE, SSE

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
        objective = GenericMCObjective(lambda Y: ((Y[...,:-1]-problem.reduction_true).pow(2)+Y[...,[-1]]).sum(dim=-1))

    # set random seed
    torch.manual_seed(trial)
    np.random.seed(trial)
    random.seed(trial)

    # random initial points and calculate intermediate outputs [cbed]
    X=draw_sobol_samples(bounds=torch.Tensor(problem.bounds),n=n_init_evals,q=1).squeeze(-2)

    # Physical images output 
    image_shape = torch.Size([X.shape[0]])+problem.measurement_true.shape
    image_output = torch.zeros(image_shape)
    for i in range(X.shape[0]):
        image_temp = torch.Tensor(problem.physics_model(X[i][0].detach().cpu().numpy(),X[i][1].detach().cpu().numpy(),X[i][2].detach().cpu().numpy())) # numpy
        image_output[i,...]=image_temp # calculate intermediate output
    if noisy:
        image_output = image_output + torch.normal(0,1,size=image_output.shape)
    
    # calculate final objective (SSE)
    SSE_value  = SSE(y_simu=image_output,y_true=problem.measurement_true,dp_pow=1).unsqueeze(-1)

    # calculate reduction intermediate outputs for EICF
    if algo=='EICF':
        y_value = torch.zeros(torch.Size([X.shape[0]])+torch.Size([1+problem.reduction_true.shape[-1]]))
        for i in range(X.shape[0]):
            y_reduction = problem.reduction_func(image_output[i,...])
            reduction_SSE = SSE(y_simu=y_reduction,y_true=problem.reduction_true,dp_pow=1)
            epsilon = torch.Tensor([SSE_value[i]-reduction_SSE])
            y_temp = torch.cat((y_reduction,epsilon),dim=-1)
            y_value[i,...]=y_temp
    obj = -1*SSE_value
    best_vals = [obj.max().detach().item()]
    best_val = obj.max()
    acqf_vals = []
    acqf_runtime = []
    for iter in range(max_iter):
        if algo in ['EI','KG','Random']:
            train_Y = SSE_value
        else:
            train_Y = y_value
        model = FixedNoiseGP(train_X=X, train_Y=train_Y, train_Yvar=torch.ones_like(train_Y) * 0.0001,
                        outcome_transform=Standardize(m=train_Y.shape[-1]),input_transform=Normalize(d=X.shape[-1]))# GP is on y, not the objective
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        start_time = time.time()
        new_x, acqf_val = get_new_sample(model=model,algo=algo,problem=problem,best_val=best_val,objective=objective)
        running_time = time.time()-start_time
        acqf_vals = acqf_vals+[acqf_val]
        acqf_runtime = acqf_runtime + [running_time]
        X = torch.cat((X,new_x),dim=0)
        image_temp=torch.Tensor(problem.physics_model(new_x[0][0].detach().cpu().numpy(),new_x[0][1].detach().cpu().numpy(),new_x[0][2].detach().cpu().numpy()))
        image_output = torch.cat((image_output,image_temp.unsqueeze(0)),dim=0)
        # calculate final objective (SSE)
        SSE_value  = torch.cat((SSE_value,SSE(y_simu=image_temp,y_true=problem.measurement_true,dp_pow=1).unsqueeze(0).unsqueeze(0)),dim=0)

        # calculate reduction intermediate outputs for EICF
        if algo=='EICF':
            y_reduction = problem.reduction_func(image_temp)
            reduction_SSE = SSE(y_simu=y_reduction,y_true=problem.reduction_true,dp_pow=1)
            epsilon = torch.Tensor([SSE_value[-1]-reduction_SSE])
            y_temp = torch.cat((y_reduction,epsilon),dim=-1).unsqueeze(0)
            y_value=torch.cat((y_value,y_temp),dim=0)
        obj = -1*SSE_value
        best_vals = best_vals+[obj.max().detach().item()]
        best_val = obj.max()
        print(f"Iteration: {iter+1}/{max_iter}")
        print(f"Suggested point: {new_x}")
        print(f"new SSE value: {SSE_value[-1]}")
        if algo not in ['EI','KG','Random']:
            print(f"Reduction valued: {y_temp}")
        print(f"Best objective function found: {-1*best_val}")
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
        new_x, acqf_val = optimize_acqf(acq_function=acqf,bounds=problem.bounds,q=1,num_restarts=20,raw_samples=100)        
        return new_x, acqf_val
    elif algo == 'EICF':
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
            torch.rand([1, problem.dim]) * (problem.bounds[1] - problem.bounds[0])
            + problem.bounds[0]
        )
        return new_x, None