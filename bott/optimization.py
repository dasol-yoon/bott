#!/usr/bin/env python3
import argparse
import logging
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

from bott.ts_acqf import ThompsonSampling
from bott.utils import time_sync, make_output_filenm, safe_division

from PIL import Image #20250520

def run_one_trial(
        problem_name: str,
        problem: SyntheticTestFunction,
        algo: str,
        trial: int,
        n_init_evals: int,
        max_iter: int,
        # metrics: List[str]=['obs_val'],
        objective: Optional[MCAcquisitionObjective] = None,
        # noisy: Optional[bool]=False,
        dtype: torch.dtype = torch.double,
        device_botorch: str='cpu',
        force_restart:Optional[bool]= False,
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
    image_dir = f"{current_directory}/results/{problem_name}/images/" #20250520
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True) #20250520
    logging.basicConfig(level=logging.INFO,  # Adjust log level as needed (DEBUG, INFO, etc.)
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)  # Get a logger for the current module
    loss_func = problem.loss_func
    reduction_true = problem.reduction_true.to(device)
    measurement_true = problem.measurement_true.to(device)
    params_abTEM = problem.params_abtem

    if objective is None:
        # objective = GenericMCObjective(lambda Y, X=None: ((Y[...,:-1]-problem.reduction_true.to(device)).pow(2)+Y[...,[-1]]).sum(dim=-1)) #TODO: The sum dimension is still a bit unclear
        objective = GenericMCObjective(lambda Y, X=None: -1*(loss_func(Y[...,:-1]*problem.scaling_factor.to(device), 
                                                                       reduction_true*problem.scaling_factor.to(device),
                                                                       reduce=True))) # should return sample_shape x batch_size x q 
        logger.info(f'Using {problem.scaling_factor} as scaling factor for the objective')

    # check if we have run the experiment. If yes, load the previous result and continue. Otherwise, start from the beginning.
    if os.path.exists(results_dir + f"trial_{trial}.pt") and not force_restart:
        logger.info(
            f"============================Resume Experiment=================================\n"
            f"Experiment: {problem_name}\n"
            f"Acquisition algo: {algo}\n"
            f"Trial seed: {trial}\n"
            f"problem.device : {problem.device}\n"
            f"GP model device: {device_botorch}"
        )
        res = torch.load(results_dir + f"trial_{trial}.pt",weights_only=False)
        # reset the random seed
        torch.set_rng_state(res["random_states"]["torch"])
        np.random.set_state(res["random_states"]["numpy"])
        random.setstate(res["random_states"]["random"])
        # Get data
        X = res["train_X"]
        train_Y = res["train_Y"]
        acqf_vals = res['acqf_val_list']
        acqf_runtime = res['acqf_runtime']
        gp_runtime = res['gp_runtime']
        physics_model_runtime = res['physics_model_runtime']
        obj = res['obj_func_val']
        pixelLoss=res['pixelLoss']
        best_val = obj.max()
        n_init_evals = X.shape[0]
        if algo == 'EICF':
            y_value = res['y_value'] 
    else:
        logger.info(
            f"============================Start New Experiment=================================\n"
            f"Experiment: {problem_name}\n"
            f"Acquisition algo: {algo}\n"
            f"Trial seed: {trial}\n"
            f"problem.device : {problem.device}\n"
            f"GP model device: {device_botorch}"
        )

        # set random seed
        torch.manual_seed(trial)
        np.random.seed(trial)
        random.seed(trial)

        # random initial points and calculate intermediate outputs [cbed]
        X=draw_sobol_samples(bounds=problem.bounds.to(device=device),n=n_init_evals,q=1).squeeze(-2).to(dtype=dtype) # X = [n_init, problem.dim], note that X is by default on cpu because problem.bounds is also on cpu

        # Physical images output 
        input_params = X.tolist()
        outputs_np = np.array([problem.get_physics_simu(*param,params_abtem_alt=params_abTEM,device_alt=problem.device) for param in input_params])
        image_output = torch.tensor(outputs_np, dtype=dtype, device=device)
        logger.info(f"image output shape {image_output.shape}")
        # if noisy: # TODO The noise on PACBED is better described by Poisson
        #     image_output = image_output + torch.normal(0,1,size=image_output.shape)
    
        # calculate final objective (Loss), this is pixelSSE. Might consider rename SSE_value into pixel_losses, and make reduction_SSE into group_loss.
        pixelLoss  = loss_func(y_simu=image_output, y_true=measurement_true, reduce=False).unsqueeze(-1) # pixelLoss = [n_init, 1]

        # calculate reduction intermediate outputs for EICF in batch
        if algo=='EICF':
            y_reduction = problem.reduction_func(image_output).to(device)  # [n_init, num_tiles]
            reductionLoss = loss_func(y_simu=y_reduction * problem.scaling_factor.to(device=device),
                                        y_true=reduction_true * problem.scaling_factor.to(device=device),
                                        reduce=False).unsqueeze(-1) # [n_init, 1]
            #epsilon = pixelLoss - reductionLoss
            epsilon = torch.Tensor(safe_division(pixelLoss, reductionLoss)).to(device=device)
            y_value = torch.cat((y_reduction,epsilon),dim=-1) # [n_init, num_tiles+1]
            
        obj = -1*pixelLoss # maximization direction
        best_val = obj.max() # tensor
        acqf_vals = []
        acqf_runtime = []
        gp_runtime = []
        physics_model_runtime =[]
    
        time_init_end = time_sync()
        logger.info(f"Initializing model with '{n_init_evals}' initial evaluations took {time_init_end - time_init_start:.3f} sec")
        logger.info("==========================================================")
    
    # Start the optimization loop
    for iter in range(n_init_evals, max_iter): # Feel like we should match the value with len(X)
        
        time_iter_start = time_sync()
        
        # Choose the acquisition function algorithm
        if algo in ['EI','KG','Random','TS']:
            train_Y = obj
        else:
            train_Y = y_value
        
        # Get model #TODO Do we really need to create model with every iteration?
        time_model_start = time_sync()
        model = SingleTaskGP(train_X=X, train_Y=train_Y, train_Yvar=None, #torch.ones_like(train_Y) * 0.0001, #TODO Need to configure this variance scaling hyperparameter
                        outcome_transform=Standardize(m=train_Y.shape[-1]),input_transform=Normalize(d=X.shape[-1])).to(device)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        time_model_end = time_sync()
        
        # Get new sample with timing
        time_sample_start = time_sync()
        new_x, acqf_val = get_new_sample(model=model,algo=algo,problem=problem,best_val=best_val,objective=objective, device=device, dtype=dtype)
        time_sample_end = time_sync()
        
        # Append and concat new values       
        acqf_vals.append(acqf_val)
        acqf_runtime.append(time_sample_end - time_sample_start)
        gp_runtime.append(time_model_end - time_init_start)
        X = torch.cat((X, new_x),dim=0)
        
        # Run physical model with a new_x
        input_param = new_x.tolist()[0] # [value0, value1, value2]
        time_simu_start = time_sync()
        image_temp = torch.from_numpy(problem.get_physics_simu(*input_param,params_abtem_alt=params_abTEM,device_alt=problem.device)).to(dtype=dtype, device=device)
        time_simu_end = time_sync()
        physics_model_runtime.append(time_simu_end-time_simu_start)
        # image_output = torch.cat((image_output, image_temp.unsqueeze(0)),dim=0) # This will continue to concat new images but image_output is never used. We should remove this unless it's needed somewhere else.
        image_t = Image.fromarray(image_temp.cpu().numpy()) #20250520
        image_t.save(image_dir+make_output_filenm(input_param)) #20250520

        # calculate final objective (Loss)
        new_Loss = loss_func(y_simu=image_temp.unsqueeze(0),y_true=measurement_true,reduce=False).unsqueeze(-1) # [1,1]
        pixelLoss = torch.cat((pixelLoss, new_Loss), dim=0)

        # calculate reduction intermediate outputs for EICF
        if algo=='EICF':
            y_reduction = problem.reduction_func(image_temp).to(device) # [num_tiles,]
            reductionLoss = loss_func(y_simu=y_reduction.unsqueeze(0)*problem.scaling_factor.to(device=device),
                                    y_true=reduction_true *problem.scaling_factor.to(device=device),
                                    reduce=False).unsqueeze(0) # [1,]
            
            # epsilon = pixelLoss[-1] - reductionLoss #20250903
            epsilon = torch.Tensor(safe_division(pixelLoss[-1], reductionLoss)).to(device=device)
            y_temp = torch.cat((y_reduction.unsqueeze(0),epsilon),dim=-1) # [1,num_tiles+1]
            logger.info(f"y_temp {y_temp}")
            y_value = torch.cat((y_value,y_temp),dim=0)
        
        # Display and save results
        obj = -1*pixelLoss
        best_val = obj.max()
        best_idx = torch.argmax(obj.squeeze(-1))
        best_params = X[best_idx].cpu().numpy()
        
        time_iter_end = time_sync()
        
        logger.info(f"\nIteration: {iter+1}/{max_iter}")
        logger.info(f"GP model fitting time: {time_model_end - time_model_start:.3f} sec")
        logger.info(f"Acqu func sampling time: {time_sample_end - time_sample_start:.3f} sec")
        logger.info(f"Physics simulation time: {time_simu_end - time_simu_start:.3f} sec")
        logger.info(f"Iteration time: {time_iter_end - time_iter_start:.3f} sec")
        logger.info(f"Suggested point: {new_x.cpu().numpy()}")
        logger.info(f"new Loss value: {new_Loss.cpu().numpy()}")
        if algo not in ['EI','KG','Random','TS']:
            logger.info(f"Reduction valued: {y_temp}")
        logger.info(f"Best iteration index found: {best_idx+1}")  
        logger.info(f"Best point found: {best_params}")
        logger.info(f"Best objective function value found: {best_val}")
        logger.info(f"==========================================================")
        if algo == 'EICF':
            BO_results = {
                "max_iter": max_iter,
                "acqf_runtime": acqf_runtime,
                "gp_runtime": gp_runtime,
                "pixelLoss":pixelLoss,
                "physics_model_runtime":physics_model_runtime,
                "acqf_val_list": acqf_vals,
                "train_X": X,
                "y_value": y_value,
                "train_Y": train_Y,
                "obj_func_val": obj,
                "random_states": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "random": random.getstate(),
                },
            }
        else:
            BO_results = {
                "max_iter": max_iter,
                "acqf_runtime": acqf_runtime,
                "gp_runtime": gp_runtime,
                "pixelLoss":pixelLoss,
                "physics_model_runtime":physics_model_runtime,
                "acqf_val_list": acqf_vals,
                "train_X": X,
                "train_Y": train_Y,
                "obj_func_val": obj,
                "random_states": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "random": random.getstate(),
                },
            }
        torch.save(BO_results, results_dir + f"trial_{trial}.pt") #save after each iteration in case of interruption
    logger.info(f"\nTotal run time with '{max_iter}' iters: {time_sync() - time_init_start:.3f} sec")

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

def parse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one replication of a BO experiment."
    )
    parser.add_argument("--trial", "-t", type=int, default=0)
    parser.add_argument("--algo", "-a", type=str, default="EI")
    parser.add_argument("--num_iter", "-n", type=int, default=50)
    
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--param_truth', "-p", type=float, nargs=3, help='Three param truth values')
    grp.add_argument('--param_truth_path', "-f", type=str, help='path to the ground truth image')

    parser.add_argument('--nt',type=int,default=1) #20250603 temporary edit
    
    args = parser.parse_args()

    # Combine into a single param_truth field
    if args.param_truth is not None:
        args.param_truth = args.param_truth  # already a list of floats
    else:
        args.param_truth = args.param_truth_path  # assign string to param_truth

    # Remove the secondary variable so main() only sees 'param_truth'
    delattr(args, 'param_truth_path')

    return args