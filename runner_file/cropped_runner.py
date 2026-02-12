#edited 20250513 cluster
import logging
import warnings

import torch
from botorch.exceptions import InputDataWarning

from bott.optimization import run_one_trial, parse
from bott.physics_models import simulate_cbed
from bott.problem import OptimizationProblem
from bott.io import load_img, load_tif
from bott.utils import print_system_info

from scipy import ndimage

logging.basicConfig(level=logging.INFO,  # Adjust log level as needed (DEBUG, INFO, etc.)
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)  # Get a logger for the current module
# logger.setLevel(logging.INFO)
# logger.handlers.pop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Device {device} is used!\n")
warnings.filterwarnings("ignore", category=InputDataWarning)
def main(
        trial: int,
        algo: str,
        num_iter: int,
        param_truth: str | list[float],
        nt = 1,
) -> None: 
        """Run one replication for the dropwave function network test problem

        Args:
            trial: Seed of the trial.
            algo: Algorithm to use. Supported algorithms: "EI", "KG", "EICF", "Random".
            num_iter: number of maximum BO iterations
            param1: Thickness
            param2: Tilt-x
            param3: Tilt-y

        Returns:
            None.
        """
        logger.info(f'domain knowledge 5 segment tile experiment. Branch Multiplied scale factor test')

        params_abTEM = {#todo: find a better way to enter parameters
                # # Device configuration
                # "device_abtem": 'gpu',#"cpu",

                # Crystal structure input
                "path_crystal": "/home/pb482/bott/one_layer_beta_GaO.cif",

                # Potential parameters
                "potential_extent_x": 48,  # Angstrom
                "potential_extent_y": 48,  # Angstrom
                "lateral_sampling": 0.2 * 2 / 3,  # Angstrom
                "vertical_sampling": 2.9,
                "potential_parametrization": "lobato",  # or "kirkland"
                "potential_projection": "finite",       # or "infinite"

                # Phonon parameters
                "random_seed": 42,
                "use_frozen_phonon": False,
                "num_phonon_configs": 5,
                "phonon_sigma": {
                    'Ga': 0.1,
                    'O': 0.1,
                },

                # Probe parameters
                "energy": 120e3,  # in eV
                "convergence_angle": 30.0,  # mrad
                "df": 0,
                "aberrations": {},

                "detector_angle": 41, #'cutoff', # mrad

                # Scan parameters
                "scan_step_size": 0.3,  # angstrom
                "return_pacbed": True,
            }
        print_system_info()
        #todo: make it into a function and put it in io.py
        if isinstance(param_truth, str):
              ground_truth = load_tif(param_truth)

              imgshape = simulate_cbed(1,0,0, params_abTEM,scan_coords=[[18,18],[30,30]]).shape
              originshape = ground_truth.shape
              #todo: the guide should ensure the experimental pacbed to be centered & square.
              ground_truth = ndimage.zoom(ground_truth, 
                                          (imgshape[0]/originshape[0], 
                                           imgshape[1]/originshape[1]), order=3)
              ground_truth = torch.Tensor(ground_truth)
              problem_name = f"domain_5seg_mult_factor_obj_adj"
              logger.info(f'max of loaded image: {torch.max(ground_truth)}')

        elif isinstance(param_truth, list):
              ground_truth = torch.Tensor(simulate_cbed(param_truth[0],param_truth[1],
                                                  param_truth[2], params_abTEM,
                                                  device_simu='gpu')) # abtem takes "cpu" or "gpu"
              problem_name = f"GT_{param_truth[0]}_{param_truth[1]}_{param_truth[2]}"
        else:
              raise ValueError("param_truth should be a list of 3 floats or a string path to the image.")

        sf_quad = 2339
        sf_cent = 6376
        temp = torch.Tensor([sf_cent, sf_quad, sf_quad, sf_quad, sf_quad])
        sf_domain5seg = torch.sqrt(temp)

        # OptimizationProblem would keep all the tensor on the specified device
        problem = OptimizationProblem(ground_truth=ground_truth,
                                    output_path='/home/pb482/bott/output_experimental/', 
                                    save_results=True, 
                                    reduction_params={'reduction_type':'domain', 'reduction_kwargs':{'radius':0.36}},
                                    loss_params={'loss_type':'SSE', 'dp_pow': 1}, 
                                    norm_arr=False,
                                    dim=3, 
                                    bounds=[(100,300), (-10, 0), (-10, 0)],
                                    noise_std=0,
                                    dtype=torch.float64, 
                                    device=device,
                                    params_abtem = params_abTEM,
                                    scale_factor=sf_domain5seg,
                                    safe_div_th_cnst = [0.2,200],
                                    scan_coords = [[18,18],[30,30]],
                                    ) # "cpu" or "cuda" for physics simulation
        logging.info(f"running 6-domain_multiplier_power_SSE.py")
        run_one_trial(problem_name=problem_name+'_41mrad', 
                    problem=problem, 
                    algo=algo, 
                    trial=trial, 
                    n_init_evals=7, 
                    max_iter=num_iter, 
                    objective=None,
                    dtype=torch.float64,
                    device_botorch=device
                    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))