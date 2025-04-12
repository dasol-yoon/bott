#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from tifffile import imread, imwrite
from torch import Tensor

from bott.io import load_tif
from bott.loss import LossFunction
from bott.physics_models import simulate_cbed
from bott.reduction import ReductionFunction
from bott.utils import make_output_filenm


class OptimizationProblem(SyntheticTestFunction):
    """ Problem Class for Hyperparameter Optimization. """
    
    def __init__(self, ground_truth: Tensor, output_path='./output', save_results=True, 
                 reduction_params: dict={'reduction_type':'square', 'reduction_kwargs':{'num_tiles':2}},
                 loss_params: dict={'loss_type': 'SSE'}, 
                 dim = 3, bounds = [(5,300), (-20, 20), (-20, 20)],
                 noise_std = None, device='cuda') -> None: # noise_std somehow has no effect when it's < 1, very weird
        self.dim = dim
        self._bounds = bounds
        super(OptimizationProblem, self).__init__(noise_std=noise_std, negate=False, bounds=self._bounds) # This has no effect unless specifically called as `get_objective(X, noisy_objective=True)`
        
        self.device = device
        self.save_results = save_results
        self.output_path = output_path
        
        # Initialize these major components
        self.measurement_true = ground_truth.to(self.device)
        self.physics_model = simulate_cbed
        self.reduction_func = ReductionFunction(reduction_params) # for optimization strategies don't need reduction (partition) we just pass None
        self.loss_func = LossFunction(loss_params,y_true=ground_truth)
        self.reduction_true = self.reduction_func(self.measurement_true)
        self.num_tiles = reduction_params['reduction_kwargs']['num_tiles']

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
            measurement_simu = self.physics_model(*X)
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
