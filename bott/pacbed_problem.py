#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The FreeSolv function network test problem.
"""


import glob
import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from abtem import *
from ase.io import read
from botorch.test_functions.synthetic import SyntheticTestFunction
from PIL import Image
from torch import Tensor


class PACBEDCalibration(SyntheticTestFunction):
    """PACBED Calibration Problem Class for Bayesian Optimization."""
    
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
        self.current_directory = os.getcwd() # Get the working directory
        self.atoms = read('/Users/pbuathong/Desktop/BoTT_data/SrTiO3.cif')  #TODO change it
        self.dx = self.atoms.cell[2, 2]
        self.fpath = '/Users/pbuathong/bott/'
        self.trueopt = np.array([[310,5,-10]])
        self.true_y = self.load(self.trueopt) #[simulate(x_val) for x_val in x_obs]
        self.partition_type = partition_type
        self.num_tiles = num_tiles
        if self.partition_type == 'domain' and self.num_tiles!=2:
            print(f"Partition type: {self.partition_type} only supports num_tiles=2.")
            print(f"Reassigning {self.num_tiles} to 2!")
            self.num_tiles=2


    
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
    
        # fp = FrozenPhonons(sc,20,{'Sr':.088,'Ti':.0746,'O':.0963},seed=1)
        potential = Potential(sc,gpts=512, projection='infinite',
                            slice_thickness= 2,
                            device='cpu', parametrization='kirkland') #TODO: to add storage = 'cpu',precalculate=True,
        probe = Probe(energy=200e3, semiangle_cutoff=19.1,tilt=(tX,tY),
                    device='cpu')
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
        imgPath = 'image/TiltX_{}_TiltY_{}_Thickness_'.format(int(tilt1),int(tilt2)) +tag
        
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
    
    def VertTiles(self,X:Tensor):
        arr = torch.Tensor([]).to(torch.double);
        ind = int(X.shape[0]//self.num_tiles)
        for i in range(ind):
            arr = torch.cat((arr, torch.mean(X[:,i*ind:(i+1)*ind]).unsqueeze(-1) *1e1), dim=-1)
        return arr
    
    def domainKnowledgeTile(self,X:Tensor):
        #create a mask to divide the image into central and peripheral regions
        ind = X.shape[0]
        mask = np.zeros((ind,ind)).astype(bool)
        mask[70:-70,70:-70] = True #central region of the image #todo systematic way: CHT?
        arr = torch.Tensor([]);
        arr = torch.cat((arr, torch.mean(X[mask]).unsqueeze(-1) *1e1), dim=-1)
        arr = torch.cat((arr, torch.mean(X[~mask]).unsqueeze(-1) *1e1), dim=-1)
        return arr
    
    def squareTiles(self, X:Tensor):
        arr = torch.Tensor([]);
        pixPerTile = int(X.shape[0]//self.num_tiles)
        for i in range(self.num_tiles):
            for j in range(self.num_tiles):
                tile = torch.mean(X[...,pixPerTile*i:pixPerTile*(i+1),...,pixPerTile*j:pixPerTile*(j+1)]).unsqueeze(-1)
                arr = torch.cat((arr,tile*1e1),dim=-1)
        return arr
    