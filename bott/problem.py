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
from bott.utils import make_output_filenm, normalize
import numpy as np


class OptimizationProblem(SyntheticTestFunction):
    """ Problem Class for Hyperparameter Optimization. """
    #default arguments
    def __init__(self, ground_truth: Tensor, 
                 output_path='./output', 
                 save_results=True, 
                 reduction_params: dict={'reduction_type':'square', 
                                         'reduction_kwargs':{'num_tiles':2}},
                 loss_params: dict={'loss_type': 'SSE'}, 
                 norm_arr = False,
                 dim = 3, 
                 bounds = [(5,300), (-20, 20), (-20, 20)],
                 noise_std = None,
                 dtype = torch.float64, 
                 device='cuda',
                 params_abtem=None,
                 scale_factor = None,
                 safe_div_th_cnst = [0.2,200],
                 scan_coords=None) -> None: # noise_std somehow has no effect when it's < 1, very weird
        self.dtype = dtype
        self.device = device
        self.norm_arr = norm_arr
        self.params_abtem = params_abtem

        self.dim = dim
        self._bounds = bounds
        super(OptimizationProblem, self).__init__(noise_std=noise_std, negate=False, bounds=self._bounds) # This has no effect unless specifically called as `get_objective(X, noisy_objective=True)`
        
        self.save_results = save_results
        self.output_path = output_path
        
        # Initialize these major components
        self.reduction_func = ReductionFunction(reduction_params) # for optimization strategies don't need reduction (partition) we just pass None
        self.loss_func = LossFunction(loss_params)
        
        self.ground_truth = ground_truth.to(dtype=self.dtype, device=self.device)
        self.measurement_true = self.get_measurement_true()
        self.reduction_true = self.get_reduction_true()
        if scale_factor is not None: #20250903
            self.scaling_factor = scale_factor.to(device=self.device)
        else:
            self.scaling_factor = self.get_scaling_factor(reduction_params).to(device=self.device) #20250908
        self.safe_div_th_cnst = safe_div_th_cnst #20250904
        self.physics_model = simulate_cbed
        if reduction_params['reduction_type'] == 'square': #20250828
            self.num_tiles = reduction_params['reduction_kwargs']['num_tiles']
        self.scan_coords = scan_coords

    def get_measurement_true(self):
        # Write it as a method so we can preprocess them in the future, like normalization, resampling and such
        if self.norm_arr:
            return normalize(self.ground_truth) #[0,1]
        else:
            return self.ground_truth
    
    def get_reduction_true(self):
        return self.reduction_func(self.get_measurement_true())
    
    def get_scaling_factor(self, rp):
        rp['reduction_kwargs'].update({'reduce': False}) # we want the full tiles, not the mean
        temp_reduction_func = ReductionFunction(rp) # update the reduction function with the new
        tiles = temp_reduction_func(self.get_measurement_true())
        del rp['reduction_kwargs']['reduce'] # remove the reduce argument, so it doesn't affect the next call
        dim = tiles[0].shape #temporary fix for list type tiles
        return torch.Tensor([np.sqrt(dim[-1]*dim[-2])]) #tile height * tile width, number of pixels in each tile
    
    def get_physics_simu(self, *params, device_alt=None, params_abtem_alt=None): 
        #allows alternative parameters different from the initial ones for testing
        # device_simu = self.device if device_alt is None else device_alt
        # device_simu = 'gpu' if device_simu == 'cuda' else 'cpu'
        if device_alt is not None:
            device_simu = device_alt #logger.info(f'case1')
        else:
            device_simu = self.device #logger.info(f'case2')
            
        device_simu = device_simu if isinstance(device_simu, str) else device_simu.type

        if device_simu in ['cuda','gpu']:
            device_simu = 'gpu'
        else:
            device_simu = 'cpu'

        param_simu_abtem = self.params_abtem if params_abtem_alt is None else params_abtem_alt
        
        simulation = self.physics_model(*params, device_simu=device_simu, 
                                        params_abtem=param_simu_abtem,
                                        scan_coords = self.scan_coords)
        if self.norm_arr:
            return normalize(simulation) #[0,1]
        else:
            return simulation
    
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
            measurement_simu = torch.from_numpy(load_tif(file_path)).to(dtype=self.dtype, device=self.device)
        else:
            device = 'gpu' if self.device == 'cuda' else None
            measurement_simu = self.physics_model(*X, device_simu=device,
                                                  scan_coords = self.scan_coords)
            if self.save_results:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                imwrite(file_path, measurement_simu)
            measurement_simu = torch.from_numpy(measurement_simu).to(dtype=self.dtype, device=self.device)
                
        measurement_true = self.get_measurement_true()
                
        # Get the loss value (objective) and return
        if self.reduction_func is not None:
            return self.loss_func(self.reduction_func(measurement_simu), self.reduction_func(measurement_true))
        else:
            return self.loss_func(measurement_simu, measurement_true)
