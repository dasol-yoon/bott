#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Problem Class for Bayesian Optimization
"""


import glob
import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from abtem import *
from ase.io import read
from PIL import Image
from torch import Tensor


class fwd_model1:
    """PACBED Calibration Problem Class for Bayesian Optimization."""
    
    def __init__(self) -> None:
        """Initialize the forward model type I.
        Returns:
            None
        """        
        # Load atoms and calculate dx
        self.current_directory = os.getcwd() # Get the working directory
        self.atoms = read('/Users/pbuathong/Desktop/BoTT_data/SrTiO3.cif')  #TODO change it
        print("x,y,z extent of the unit cell ($\AA$): ",self.atoms.cell)
        self.dx = self.atoms.cell[2, 2]
        print('Thickness step: ',self.dx,' ($\AA$)')
        self.fpath = '/Users/pbuathong/bott/'
    
    def simulate(self,x,tX,tY):
        #----variables----
        #x: thickness in angstrom
        #start = datetime.now()
        
        numUC = int(x/self.dx)#number of unit cells in z direction (variable x)
        #tag = '/vac_conv19p1_volt200kV_tilt0each_{:04d}.tiff'.format(int(numUC*dx))
        sc = self.atoms * (16,16,numUC) #todo: may need to do the rounding instead
        #sc.center(axis=(0,1),vacuum=10)
        
        #fp = FrozenPhonons(sc,20,{'Sr':.1,'Ti':.1,'O':.1},seed=1)
        fp = FrozenPhonons(sc,20,{'Sr':.088,'Ti':.0746,'O':.0963},seed=1)
        #potential = Potential(fp,gpts=512, projection='infinite',# ###comment: original line
        potential = Potential(sc,gpts=512, projection='infinite',# ###comment: line changed from fp -> sc for fast testing for cpu only
                            #sampling=0.15, projection='infinite', #,sampling=0.06
                            slice_thickness= 2,storage  = 'cpu', precalculate=True,
                            device='cpu', parametrization='kirkland')
        probe = Probe(energy=200e3, semiangle_cutoff=19.1,tilt=(tX,tY),
                    device='cpu')
        probe.grid.match(potential)
        # print(probe.cutoff_scattering_angles)
        
        pixelated_detector = PixelatedDetector(max_angle=45) #temporary limit for thickness
    #29.345 33.258 
        gridscan = GridScan(start=[29.345,29.345], end=[33.258 ,33.258],sampling=.3)
        #scan = GridScan(start=[19.7,19.7], end=[27.6,27.6],sampling=.2)

        pixelated_measurement = probe.scan(gridscan, pixelated_detector, potential,pbar=False)
        pacbed = np.mean(pixelated_measurement.array,axis=(0,1)).astype(np.float32)
        
        #end = datetime.now() 
        # print('Time elapsed (hh:mm:ss.ms):  {}  || thickness: {} $\AA$  || shape: {}'.format(end-start,
        #                                                                                     numUC*dx,
                                                                                            # pacbed.shape))
        
        return pacbed
        
    def load(self,x):
        #----variables----
        #x: [[thickness [angstrom], tilt]]
        #returns pre-simulated PACBED image.
        thickness=x[0,0].item()
        tilt1 = x[0,1].item()   #20250312: make it into int here?
        tilt2 = x[0,2].item()   

        numUC = int(thickness/self.dx)#number of unit cells in z direction (variable x)
        tag = '{:04d}.tiff'.format(int(numUC*self.dx))
        imgPath = 'TiltX_{}_TiltY_{}_Thickness_'.format(int(tilt1),int(tilt2)) +tag
        print(f"imgPath {imgPath}")
        
        try: #load PACBED from the directory
            pacbed = plt.imread(imgPath)
            pacbed = pacbed.astype(np.float32) #added 20241112. bad simulation. crop zeroes.
        except: #simulate
            #print('no such file: ' +imgPath)
            pacbed = self.simulate(thickness,tilt1,tilt2)
            pacbedImg = Image.fromarray(pacbed)
            pacbedImg.save(self.fpath+imgPath)

        #pacbed_sqrt = np.sqrt(pacbed) #added 20241017
        #pacbed_norm = pacbed_sqrt/np.mean(pacbed_sqrt) #added 20241017; "standardize" the data...? Isn't it redundant with the normalization?
        pacbed_norm = (pacbed-np.min(pacbed))/(np.max(pacbed)-np.min(pacbed)) #normalize 0-1
        #pacbed_smol = scipy.ndimage.zoom(pacbed_norm.astype(np.float64),20/pacbed_norm.shape[0],order=1) #Nov12: comment out
        #pacbed_norm = (pacbed_smol-np.min(pacbed_smol))/(np.max(pacbed_smol)-np.min(pacbed_smol)) #normalize 0-1 #Nov12: comment out
        

        #unicode for angstrom
        #print('thickness: {} \u212B  || shape: {} || filename: '.format(x, pacbed_norm.shape)+imgPath[90:])
        
        return torch.Tensor(pacbed_norm)#.reshape(-1) #PB: change to tensor
 

        
