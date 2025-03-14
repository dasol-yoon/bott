# Put the data loading and saving functions here
# Exp PACBED, cif, params files

import numpy as np
import torch
import glob
import matplotlib.pyplot as plt

#todo: group the functions based on the functionalities so that they are easy to
# locate and import
# e.g. data loading, visualization, models, etc.
# utils are for helper functions that don't fit elsewhere
# could also have setup.py

tkwargs = {"dtype":torch.double}

def load(x, filePath, crop=None):
    '''
    returns pre-simulated PACBED image specified by the variable x
    or simulate on the fly
    
    ----variables----
    x: ndarray of shape (1,3)
        [ [thickness (angstrom), tiltX, tiltY] ] 

    filePath = file path where simulated images will be/are saved

    crop: integer that  will crop the diffraction patterns by
        DP[ crop:-crop, crop: -crop]
        if the sample is too noisy or not well simulated, crop out area w/ less info

    todo: add the abtem simulation version
    todo: consider object oriented approach for better error handling?
    todo: might consider dividing this function into multiple functions

    todo: error handling - don't let cropping go beyond 1/2 of the image size
    '''
    thickness=x[0,0]
    tilt = x[0,1]   
    
    tag = '{:04d}'.format(int(thickness)) #format the thickness value for file name

    #todo: need to test and finalize the folder structure.
    #probably it's better to include tilt in a file name for continuous tilt guesses
    #todo: currently placeholder path for testing. Need to change to the right path
    imgPath = glob.glob(filePath+'TiltX_{}_TiltY_0_Thickness_'.format(int(tilt))
                        +tag[:3]+'*.tif')[0] #1st in list
    
    try: #load PACBED from the directory
        y = plt.imread(imgPath)
    except FileNotFoundError: #handle exceptions
        raise ValueError('Check the file path again. There is no such file: ' +imgPath)
    #todo: need to add the case for simulation.
    #todo: need to check whether the filepath is correct first
    except Exception as e:
        raise ValueError('Something went wrong while loading: ', e)
    
    if crop is not None: #user defined something
        if not isinstance(crop,int): #not an integer
            raise ValueError('Crop should be an integer if defined')
        y = y[crop:-crop,crop:-crop].astype(np.float64) 
    else: #no cropping
        y = y.astype(np.float64)
    
    #pacbed_sqrt = np.sqrt(pacbed)
    #pacbed_norm = pacbed_sqrt/np.mean(pacbed_sqrt) # "standardize" the data...? Isn't it redundant with the normalization?
    y_full = (y-np.min(y))/(np.max(y)-np.min(y)) #normalize 0-1
    
    return torch.Tensor(y_full)