#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Problem Class for Bayesian Optimization
"""
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from abtem import *
from ase.io import read
from botorch.test_functions.synthetic import SyntheticTestFunction
from PIL import Image
from torch import Tensor

from bott.forward_models import fwd_model1
from bott.loss import *
from bott.metrics import *


class ElectronMicroscopyCalibration(SyntheticTestFunction):
    """Electron Microscopy Calibration Problem Class for Bayesian Optimization."""
    
    def __init__(self, true_opt: Tensor, fwd_model_type: str='default', partition_type: str='square', num_tiles: int=9, loss:str='SSE', **kwargs) -> None:
        """Initialize the function network.

        Args:
            fwd_model_type: A str indicating the model class
                Options:
                            'default': default model
            partition_type: A str specifying type of partition to be used. 
                Options:    'vert': partiioning vertically, 
                            'square': partitioning as square grids, 
                            'domain': partitioning using domain knowledge 
            num_tiles: A int specifying number of tiles to be partitioned
            true_opt: A tensor containing the true parameter

        Returns:
            None
        """      
        self.fwd_model_type = fwd_model_type  
        self.num_tiles = num_tiles
        self.partition_type = partition_type
        self.loss = loss
        self.true_opt  = true_opt
        # Load forward model
        if self.fwd_model_type == 'default':
            self.fwd_model = fwd_model1()
        
        # Load tiling function
        if self.partition_type == 'vert':
            self.tiling = VertTile(num_tiles=self.num_tiles)
        elif self.partition_type == 'domain':
            if self.num_tiles !=2:
                raise ValueError(f"Domain Knowledge Tiling only supports num_tile = 2, but got {self.num_tiles}")
            self.tiling = domainKnowledgeTile(num_tiles=self.num_tiles) 
        elif self.partition_type == 'square':
            self.tiling = squareTile(num_tiles=self.num_tiles)
        else:
            raise ValueError(f"The current implementation does not support tiling type {self.partition_type}")
        if self.loss == 'SSE':
            self.loss_cal = tileSSE
        self.tile_evaluate = self.tiling.evaluate_tile
        self.y_true = self.fwd_model.load(x=np.array(self.true_opt))

    def evaluate_true(self, X: Tensor) -> None:
        return None
    
    def evaluate(self,X: Tensor):
        output = torch.zeros(X.shape[0],1).to(torch.double)
        print(f"outputshape {output.shape}")
        for i in range(X.shape[0]):
            x_array=np.array(X[[i],...])
            y_sim = self.fwd_model.load(x=x_array)
            print(f"shape y_sim {y_sim.shape}")
            tile_x = self.tile_evaluate(y_sim)
            print(f"tile_x shape {tile_x.shape}")
            print(f"y_true shape {self.y_true}")
            output[i,0]=self.loss_cal(tiles1=tile_x,tiles2=self.y_true)
        return output
