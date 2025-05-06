import logging
import warnings

import torch
from botorch.exceptions import InputDataWarning

from bott.optimization import parse, run_one_trial
from bott.physics_models import simulate_cbed
from bott.problem import OptimizationProblem

logging.basicConfig(level=logging.INFO,  # Adjust log level as needed (DEBUG, INFO, etc.)
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)  # Get a logger for the current module
# logger.setLevel(logging.INFO)
# logger.handlers.pop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Device {device} is used!")
warnings.filterwarnings("ignore", category=InputDataWarning)
def main(
        trial: int,
        algo: str,
        num_iter: int,
        param_truth: list[float]
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
        ground_truth = torch.Tensor(simulate_cbed(param_truth[0],param_truth[1],param_truth[2], device_simu='gpu')) # abtem takes "cpu" or "gpu"
        # OptimizationProblem would keep all the tensor on the specified device
        problem_name = f"GT_{param_truth[0]}_{param_truth[1]}_{param_truth[2]}"
        problem = OptimizationProblem(ground_truth=ground_truth,
                                    output_path='./output', 
                                    save_results=True, 
                                    reduction_params={'reduction_type':'square', 'reduction_kwargs':{'num_tiles':2}},
                                    loss_params={'loss_type':'SSE', 'dp_pow': 0.5}, 
                                    norm_arr=False,
                                    dim=3, 
                                    bounds=[(5,300), (-20, 20), (-20, 20)],
                                    noise_std=0,
                                    dtype=torch.float64, 
                                    device=device
                                    ) # "cpu" or "cuda" for physics simulation

        run_one_trial(problem_name=problem_name, 
                    problem=problem, 
                    algo=algo, 
                    trial=trial, 
                    n_init_evals=2, 
                    max_iter=num_iter, 
                    objective=None,
                    dtype=torch.float64,
                    device_botorch=device
                    )


if __name__ == "__main__":
    args = parse()
    main(**vars(args))