#edited 20250513 cluster
import sys
from datetime import datetime
from pathlib import Path
import random
import numpy as np
# Ensure project root is on PYTHONPATH so "bott" can be imported when run from run_job/ or via SLURM
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging
import warnings
from bott.utils import add_poisson_noise
import torch
from botorch.exceptions import InputDataWarning

from bott.optimization import run_one_trial, parse
from bott.physics_models import simulate_cbed
from bott.problem import OptimizationProblem
from bott.io import load_img
from bott.utils import print_system_info

from scipy import ndimage

logging.basicConfig(level=logging.INFO,  # Adjust log level as needed (DEBUG, INFO, etc.)
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)  # Get a logger for the current module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Device {device} is used!\n")
warnings.filterwarnings("ignore", category=InputDataWarning)
def main(
        trial: int,
        algo: str,
        num_iter: int,
        n_init_evals: int,
        param_truth: str | list[float],
        noisy_ground_truth_peak: float | None = None,
        manual_init_evals: list[list[float]] | None = None,
) -> None: 
        """Run one replication for simulated experiment

        Args:
            trial: Seed of the trial.
            algo: Algorithm to use. Supported algorithms: "EI", "KG", "EICF", "Random".
            num_iter: number of maximum BO iterations
            n_init_evals: number of initial evaluations
            param_truth: ground truth parameters [thickness, tilt-x, tilt-y]
            noisy_ground_truth_peak: max photon count (controls noise level)
            manual_init_evals: list of lists of initial evaluation points
        Returns:
            None.
        """
        #TODO here are parameters to change for different experiments
        overall_scaling_factor = 1000
        eps_bound = 10
        run_date = datetime.today().strftime("%Y-%m-%d") #22/04/2026 for new composite form
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f'--------------------------------------------------------------------------------')
        logger.info(f'Running simulated experiments with configurations below:')
        logger.info(f'Ground truth: {param_truth}')
        logger.info(f'Noisy ground truth peak: {noisy_ground_truth_peak}')
        logger.info(f'Run date: {run_date}')
        logger.info(f'Seed: {seed}')
        logger.info(f'Overall scaling factor: {overall_scaling_factor}')
        logger.info(f'Epsilon bound: {eps_bound}')
        logger.info(f'Algorithm: {algo}')
        logger.info(f'Number of iterations: {num_iter}')
        logger.info(f'Number of initial evaluations: {n_init_evals}')
        logger.info(f'Noisy ground truth peak: {noisy_ground_truth_peak}')
        logger.info(f'--------------------------------------------------------------------------------')


        params_abTEM = {

                # Crystal structure input
                "path_crystal": "/home/pb482/bott/data/SrTiO3.cif",

                # Potential parameters
                "potential_extent_x": 62.6,  # Angstrom
                "potential_extent_y": 62.6,  # Angstrom
                "lateral_sampling": 0.2 * 2 / 3,  # Angstrom
                "vertical_sampling": 2,
                "potential_parametrization": "lobato",  # or "kirkland"
                "potential_projection": "finite",       # or "infinite"

                # Phonon parameters
                "random_seed": 42,
                "use_frozen_phonon": False,
                "num_phonon_configs": 5,
                "phonon_sigma": {
                    'Sr': 0.088,
                    'Ti': 0.0746,
                    'O': 0.0963,
                },

                # Probe parameters
                "energy": 200e3,  # in eV
                "convergence_angle": 19.1,  # mrad
                "df": 0,
                "aberrations": {},

                "detector_angle": 31, #'cutoff', # mrad

                # Scan parameters
                "scan_step_size": 0.3,  # angstrom
                "return_pacbed": True,
            }
        print_system_info()

        #todo: make it into a function and put it in io.py
        if isinstance(param_truth, str):
            ground_truth = load_img(param_truth)

            imgshape = simulate_cbed(1,0,0, params_abTEM).shape
            originshape = ground_truth.shape
            #todo: the guide should ensure the experimental pacbed to be centered & square.
            ground_truth = ndimage.zoom(ground_truth, 
                                        (imgshape[0]/originshape[0], 
                                        imgshape[1]/originshape[1]), order=3)
            ground_truth = torch.Tensor(ground_truth)
            # ground_truth = (ground_truth / ground_truth.sum())*overall_scaling_factor #3/18/2026 added overall scaling factor to scale up the image output
            logger.info(f"Scaling ground truth by overall scaling factor: {overall_scaling_factor}")
            ground_truth = ground_truth*overall_scaling_factor
            problem_name = f"domain_5seg_mult_factor_obj_adj_newcomposite_eps_bound_{eps_bound}"

        elif isinstance(param_truth, list):
            ground_truth = torch.Tensor(simulate_cbed(param_truth[0],param_truth[1],
                                                  param_truth[2], params_abTEM,
                                                  device_simu='gpu')) # abtem takes "cpu" or "gpu"
            # ground_truth = (ground_truth / ground_truth.sum())*overall_scaling_factor #3/18/2026 added overall scaling factor to scale up the image output
            logger.info(f"Scaling ground truth by overall scaling factor: {overall_scaling_factor}")
            ground_truth = ground_truth*overall_scaling_factor
            problem_name = f"GT_{param_truth[0]}_{param_truth[1]}_{param_truth[2]}_newcomposite_eps_bound_{eps_bound}"
        else:
              raise ValueError("param_truth should be a list of 3 floats or a string path to the image.")
        if noisy_ground_truth_peak is not None and noisy_ground_truth_peak > 0:
            ground_truth_original = ground_truth.clone()
            logger.info(f"Before adding noise, ground_truth (max, min) : ({torch.max(ground_truth)}, {torch.min(ground_truth)}) with shape {ground_truth.shape}")
            ground_truth = add_poisson_noise(image=ground_truth, peak=noisy_ground_truth_peak)
            logger.info(f"After adding noise, ground_truth (max, min) : ({torch.max(ground_truth)}, {torch.min(ground_truth)}) with shape {ground_truth.shape}")
            is_noisy_ground_truth = f"noisy_{noisy_ground_truth_peak}"
            problem_name = problem_name + f"_noisy_{noisy_ground_truth_peak}"
        else:
            ground_truth_original = ground_truth.clone()
            is_noisy_ground_truth = "nonoise"
            problem_name = problem_name + f"_nonoise"
        sf_quad = 4303
        sf_cent = 7440
        temp = torch.Tensor([sf_cent, sf_quad, sf_quad, sf_quad, sf_quad])
        sf_domain5seg = torch.sqrt(temp)

        # OptimizationProblem would keep all the tensor on the specified device
        problem = OptimizationProblem(ground_truth=ground_truth,
                                    output_path='/home/pb482/bott/output/', 
                                    save_results=True, 
                                    reduction_params={'reduction_type':'domain', 'reduction_kwargs':{'radius':0.31}},
                                    loss_params={'loss_type':'SSE', 'dp_pow': 1}, 
                                    norm_arr=False,
                                    dim=3, 
                                    bounds=[(5,500), (-10, 10), (-10, 10)],
                                    noise_std=0,
                                    dtype=torch.float64, 
                                    device=device,
                                    params_abtem = params_abTEM,
                                    scale_factor=sf_domain5seg,
                                    safe_div_th_cnst = [0.2,200],
                                    overall_scaling_factor = overall_scaling_factor
                                    ) # "cpu" or "cuda" for physics simulation
        if manual_init_evals is not None:
            run_one_trial(problem_name=problem_name+'_SSE_dppow1_noNorm_init_'+str(n_init_evals)+'_manual_'+str(len(manual_init_evals))+'_'+is_noisy_ground_truth+'_overall_scale_factor_'+str(overall_scaling_factor)+'_run_date_'+run_date, 
                    problem=problem, 
                    algo=algo, 
                    trial=trial, 
                    n_init_evals=n_init_evals, 
                    max_iter=num_iter, 
                    objective=None,
                    dtype=torch.float64,
                    device_botorch=device,
                    manual_init_evals = manual_init_evals,
                    ground_truth_original = ground_truth_original,
                    eps_bound = eps_bound,
                    )
        else:
            run_one_trial(problem_name=problem_name+'_SSE_dppow1_noNorm_init_'+str(n_init_evals)+'_'+is_noisy_ground_truth+'_overall_scale_factor_'+str(overall_scaling_factor)+'_run_date_'+run_date, 
                    problem=problem, 
                    algo=algo, 
                    trial=trial,
                    n_init_evals=n_init_evals, 
                    max_iter=num_iter, 
                    objective=None,
                    dtype=torch.float64,
                    device_botorch=device,  
                    ground_truth_original = ground_truth_original,
                    eps_bound = eps_bound,
                    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))