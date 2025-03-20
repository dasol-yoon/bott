#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The FreeSolv function network test problem.
"""


import glob
import os
import warnings
from datetime import datetime
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from abtem import *
from ase.io import read
from botorch.test_functions.synthetic import SyntheticTestFunction
from PIL import Image
from torch import Tensor


class PECBEDCalibration(SyntheticTestFunction):
    """PECBED Calibration Problem Class for Bayesian Optimization."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize the function network.

        Args:
            node_costs: cost of evaluating each of the nodes in the function network.

        Returns:
            None
        """        
        # Load atoms and calculate dx
        self.current_directory = os.getcwd() # Get the working directory
        self.atoms = read('/Users/pbuathong/Desktop/BoTT_data/SrTiO3.cif')  #TODO change it
        self.dx = self.atoms.cell[2, 2]
        self.fpath = '/Users/pbuathong/bott/results'

    
    def simulate(self,x,tX,tY):
        '''
        Args: 
            - x: thickness in angstrom
            - tX: tilt on x-axis
            - tY: tilt on y-axis
        Returns:
            - PECBED Image
        '''

        numUC = int(x/self.dx) # number of unit cells in z direction (variable x)
        sc = self.atoms * (16,16,numUC) #TODO: may need to do the rounding instead
    
        fp = FrozenPhonons(sc,20,{'Sr':.088,'Ti':.0746,'O':.0963},seed=1)
        potential = Potential(fp,gpts=512, projection='infinite',
                            slice_thickness= 2,
                            device='gpu', parametrization='kirkland') #TODO: to add storage = 'cpu',precalculate=True,
        probe = Probe(energy=200e3, semiangle_cutoff=19.1,tilt=(tX,tY),
                    device='gpu')
        probe.grid.match(potential)
    
        pixelated_detector = PixelatedDetector(max_angle=45)
        gridscan = GridScan(start=[29.345,29.345], end=[33.258 ,33.258],sampling=.3)

        pixelated_measurement = probe.scan(gridscan, pixelated_detector, potential) #TODO: to add pbar=False
        pacbed = np.mean(pixelated_measurement.array,axis=(0,1)).astype(np.float32)
        return pacbed
    
    def load(self,x):
        '''
        Args: 
            - x: two dimensions array in the format [[thickness[angstrom],tilt]]
        Returns:
            - pre-simulated PACBED image
        '''
        thickness=x[0,0].item()
        tilt1 = x[0,1].item()   #20250312: make it into int here?
        tilt2 = x[0,2].item()   

        numUC = int(thickness/self.dx)#number of unit cells in z direction (variable x)
        tag = '{:04d}.tiff'.format(int(numUC*self.dx))
        imgPath = 'TiltX_{}_TiltY_{}_Thickness_'.format(int(tilt1),int(tilt2)) +tag
        
        try: #load PACBED from the directory
            pacbed = plt.imread(imgPath)
            pacbed = pacbed.astype(np.float32) #added 20241112. bad simulation. crop zeroes.
        except: #simulate
            pacbed = self.simulate(thickness,tilt1,tilt2)
            pacbedImg = Image.fromarray(pacbed)
            pacbedImg.save(self.fpath+imgPath)

        pacbed_norm = (pacbed-np.min(pacbed))/(np.max(pacbed)-np.min(pacbed)) #normalize 0-1
        
        return torch.Tensor(pacbed_norm)

    def evaluate_true(self, X: Tensor) -> None:
        return None
    
    def evaluate(self,):
        return None
    