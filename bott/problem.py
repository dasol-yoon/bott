#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The FreeSolv function network test problem.
"""

from typing import List, Optional, Union
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction


class OptimizationProblem(SyntheticTestFunction):
    """Problem Class for Hyperparameter Optimization."""
    
    def __init__(self, partition_type: str='square', num_tiles: int=9, **kwargs) -> None:
        """Initialize the function network.

        Args:
            partition_type: A str specifying type of partition to be used. 
                Options:    'vert': partiioning vertically, 
                            'square': partitioning as square grids, 
                            'domain': partitioning using domain knowledge 
            num_tiles: A int specifying number of tiles to be partitioned

        Returns:
            None
        """        
        # Load atoms and calculate dx
        # self.current_directory = os.getcwd() # Get the working directory
        # self.atoms = read('/Users/pbuathong/Desktop/BoTT_data/SrTiO3.cif')  #TODO change it
        # self.dx = self.atoms.cell[2, 2]
        # self.fpath = '/Users/pbuathong/bott/'
        # self.trueopt = np.array([[310,5,-10]])
        
        # Initialize these major components
        self.physics_model = None
        self.reduction_func = None # for optimization strategies don't need reduction (partition) we just pass None
        self.loss_func = None
        
        # self.true_y = self.load(self.trueopt) #[simulate(x_val) for x_val in x_obs]
        # self.partition_type = partition_type
        # self.num_tiles = num_tiles
        # if self.partition_type == 'domain' and self.num_tiles!=2:
        #     print(f"Partition type: {self.partition_type} only supports num_tiles=2.")
        #     print(f"Reassigning {self.num_tiles} to 2!")
        #     self.num_tiles=2

    def get_objective(self, X, noisy=False):
        """
        wrapper function to return objective
        """
        # __call__(X) can be configured by BoTorch to add noise on the objective
        # evaluate_true(X) is used to get objective directly from the physics mdel
        
        return self.__call__(X) if noisy else self.evaluate_true(X)

    def evaluate_true(self, X):
        """
        Return the objective by combining loss, tiling, and physics model 
        """
        if self.reduction_func is not None:
            return self.loss_func(self.reduction_func(self.physics_model(X)))
        else:
            return self.loss_func(self.physics_model(X))