#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from tifffile import imread, imwrite

from bott.io import load_tif
from bott.physics_models import simulate_cbed
from bott.utils import make_output_filenm
from bott.reduction import ReductionFunction
from bott.loss import LossFunction


class OptimizationProblem(SyntheticTestFunction):
    """ Problem Class for Hyperparameter Optimization. """
    
    def __init__(self, ground_truth_path: str, output_path='/output', save_results=True, 
                 reduction_params: dict={'reduction_type':'square', 'num_tiles':2},
                 loss_params: dict={'loss_type': 'SSE'}, 
                 noise_std = None, device='cuda', **kwargs) -> None:
        self.device = device
        self.save_results = save_results
        self.output_path = output_path
        super().__init__(noise_std=noise_std) # This has no effect unless specifically called as `get_objective(X, noisy_objective=True)`
        
        # Initialize these major components
        self.measurement_true = torch.from_numpy(imread(ground_truth_path)).to(self.device)
        self.physics_model = simulate_cbed
        self.reduction_func = ReductionFunction(reduction_params) # for optimization strategies don't need reduction (partition) we just pass None
        self.loss_func = LossFunction(loss_params)

    def get_measurement_true(self):
        # Write it as a method so we can preprocess them in the future, like normalization, resampling and such
        return self.measurement_true
    
    def get_objective(self, X, noisy_objective=False):
        """
        wrapper function to return objective
        """
        # __call__(X) is implemented by BoTorch and it can do noising / transformation on self.evaluate_true(X)
        # evaluate_true(X) is used to get objective directly from the physics model. This naming is BoTorch convention.
        
        return self.__call__(X) if noisy_objective else self.evaluate_true(X)

    def evaluate_true(self, X):
        """
        Return the objective by combining loss, tiling, and physics model 
        """
        
        # Try to get measurement_simu from file, if not then simulate
        file_path = os.path.join(self.output_path, make_output_filenm(X))
        if os.path.exists(file_path):
            measurement_simu = torch.from_numpy(load_tif(file_path)).to(self.device)
        else:
            measurement_simu = self.physics_model(X)
            if self.save_results:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                imwrite(file_path, measurement_simu)
            measurement_simu = torch.from_numpy(measurement_simu).to(self.device)
                
        measurement_true = self.get_measurement_true()
                
        # Get the loss value (objective) and return
        if self.reduction_func is not None:
            return self.loss_func(self.reduction_func(measurement_simu), self.reduction_func(measurement_true))
        else:
            return self.loss_func(measurement_simu, measurement_true)
