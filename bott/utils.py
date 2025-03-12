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

def getNumPix(y_cand_full, numTiles): 
    #returns the number of pixels within one tile
    #currently only support square tiles
    return y_cand_full.shape[0]//numTiles

#todo implement patches with overlaps/non uniform size/etc.
def intoTiles(y_cand_full, numTiles, scalingFactor=1e1):
    '''
    returns a tensor of size (numTiles**2)
        the mean values of each image tile, multipled by the scaling factor
    '''
    arr = torch.Tensor([])
    pixPerTile = getNumPix(y_cand_full,numTiles) #number of pixels within one patch
    for i in range(numTiles):
        for j in range(numTiles):
            tile = torch.mean(y_cand_full[...,pixPerTile*i:pixPerTile*(i+1),...,pixPerTile*j:pixPerTile*(j+1)]).unsqueeze(-1)
            arr = torch.cat((arr,tile*scalingFactor),dim=-1)
    return arr 

def pixelSSE(y_cand_full, y_target_full):
    err = y_cand_full-y_target_full
    return err.pow(2).sum().unsqueeze(-1) #np.power(err,2).sum().unsqueeze(-1) 

def tileSSE(tiles1, tiles2): #SSE of patches (candidate vs. reference)
    #todo: need error handling for the case when the two tensors have different shapes
    return (tiles1-tiles2).pow(2).sum().unsqueeze(-1)

def computeY(y_cand_full, y_target_full, numTiles):
    '''
    return variable: torch.Tensor of size [1, numTiles**2 +1]
    '''
    y_cand_tile = intoTiles(y_cand_full, numTiles)

    #compute the error term.
    y_target_tile = intoTiles(y_target_full, numTiles) #reference
    tileTerm = tileSSE(y_cand_tile,y_target_tile) #single value based on tile
    pixelTerm = pixelSSE(y_cand_full, y_target_full) #single value based on pixel
    return torch.cat((y_cand_tile,pixelTerm-tileTerm),dim=-1).unsqueeze(0) #y_pred

# def computeObjectiveC(y_pred): #TODO: using pixelSSE(y_pred_raw) may enhance the speed a bit?
#     return -((y_pred[...,:-1]-y_obsC[...,:-1]).pow(2).sum()+y_pred[...,-1]).unsqueeze(-1)
#     # return - (np.power( y_pred[:9]-y_obsC[:9], 2).sum() + y_pred[9] ) #pixelTerm - patchTerm #question: isn't it just the pixelTerm?

# todo: consider using classicBO as a class and compositeBO as a class
# then have add_data as a method for each class
#todo: filePath can be initialized in the __init__ method
#or use environment variable
def add_data_composite(new_x, filePath, y_target_full, numTiles, x=None, y=None, obj=None,tkw = tkwargs):
    '''
    tkw: dictionary of torch keyword arguments. If the images are 
    huge(> 1000x1000 pixels), gpu could be used, but performance of BO 
    is not guaranteed to be better with GPU. e.g. {"dtype":torch.double,
    "device": torch.device('cpu')}
    '''
    if x is None:
        x = torch.tensor([],**tkw)
    if y is None:
        y = torch.tensor([],**tkw)
    if obj is None:
        obj = torch.tensor([],**tkw)

    new_y_full = load(new_x, filePath)
    new_y = computeY(new_y_full,y_target_full,numTiles) # tiles
    new_obj = - pixelSSE(new_y_full, y_target_full) #SSE

    # x and obj are both 2D tensors that are nx1
    # y is a 3D tensor that is nx (numPatches**2 + 1) 
    x = torch.cat((x, new_x),dim=0)
    y = torch.cat((y, new_y),dim=0)
    obj = torch.cat((obj, torch.tensor(new_obj.clone().detach())),dim=0)
    return x.to(**tkw), y.to(**tkw), obj.to(**tkw)

from botorch.sampling.normal import IIDNormalSampler #import package
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import Normalize

def generate_initial_data(n_initial_points,lower, upper, filePath, y_target_full, numTiles,tkw=tkwargs):    
    new_x = (
            draw_sobol_samples(
                bounds=torch.tensor([[lower,1], [upper,10]]).to(torch.double),
                n=n_initial_points,
                q=1,
            )
        ).to(**tkw)
    for i in range(n_initial_points):
        if i == 0:
            x,y,obj = add_data_composite(new_x[i,...],filePath,y_target_full,numTiles)
        else:
            x,y,obj = add_data_composite(new_x[i,...],filePath,y_target_full,numTiles,x,y,obj)
    return x,y,obj


#todo: result = run_pipeline(y_target, method="tile") #patch, etc. 
def run_optimization(y_target, filePath, numTiles, lower, upper, n_BO_points=30, n_initial_points=4, tkw=tkwargs):
    '''
    y_target: the target PACBED image
    filePath: the path where the simulated PACBED images are saved
    numTiles: number of tiles to divide the image
    n_steps: number of optimization steps
    n_initial_points: number of initial points
    lower: lower bound for the thickness
    upper: upper bound for the thickness
    tkw: torch keyword arguments
    '''
    x,y,obj = generate_initial_data(n_initial_points,lower, upper, filePath, y_target, numTiles, tkw)
    


#todo: plotting functions in a separate file

# def initY(path, testModeX=None):#image in question. 
#     '''
#     turn the testModeX on to test the pacakge with a set of X values
#     by defining them your own. testModeX = x
#     example: initY( np.array( [[250,4,0]] ))
#     '''
#     if testModeX:
#         #toodo: need to check the format and shape
#         y_raw = load(testModeX)
#     else:
#         print()
#         #todo: define a function to format the text image to match the template
    
#     y_patC = intoTiles(y_raw) # compute reference values for patches
#     y_obsC = computeY(y_raw) # compute reference values
#     return y_raw, y_patC, y_obsC



#todo: set lower and upper limit when it runs