## Put the loss functions and reduction methods
import numpy as np
import torch
from torch import Tensor

tkwargs = {"dtype":torch.double}

# Reduction methods
class VertTile:
    def __init__(self, num_tiles: int=9, **kwargs) -> None:
        self.num_tiles = num_tiles
    def evaluate_tile(self,X:Tensor):
        arr = torch.Tensor([]).to(torch.double)
        ind = int(X.shape[0]//self.num_tiles)
        for i in range(ind):
            arr = torch.cat((arr, torch.mean(X[:,i*ind:(i+1)*ind]).unsqueeze(-1) *1e1), dim=-1)
        return arr

class domainKnowledgeTile:
    def __init__(self, num_tiles: int=2, **kwargs) -> None:
        self.num_tiles = num_tiles
    def evaluate_tile(self,X:Tensor):
        #create a mask to divide the image into central and peripheral regions
        ind = X.shape[0]
        mask = np.zeros((ind,ind)).astype(bool)
        mask[70:-70,70:-70] = True #central region of the image #todo systematic way: CHT?
        arr = torch.Tensor([])
        arr = torch.cat((arr, torch.mean(X[mask]).unsqueeze(-1) *1e1), dim=-1)
        arr = torch.cat((arr, torch.mean(X[~mask]).unsqueeze(-1) *1e1), dim=-1)
        return arr

class squareTile:
    def __init__(self, num_tiles: int=2, **kwargs) -> None:
        self.num_tiles = num_tiles
    def evaluate_tile(self, X:Tensor):
        arr = torch.Tensor([])
        pixPerTile = int(X.shape[0]//self.num_tiles)
        for i in range(self.num_tiles):
            for j in range(self.num_tiles):
                tile = torch.mean(X[...,pixPerTile*i:pixPerTile*(i+1),...,pixPerTile*j:pixPerTile*(j+1)]).unsqueeze(-1)
                arr = torch.cat((arr,tile*1e1),dim=-1)
        return arr


# OLD Code
# ## Put the loss functions and reduction methods
# import torch
# import numpy as np

# tkwargs = {"dtype":torch.double}

# # Reduction methods
# def twoVertTiles(y_raw):
#     arr = torch.Tensor([])
#     arr = torch.cat((arr, torch.mean(y_raw[:,:75]).unsqueeze(-1) *1e1), dim=-1)
#     arr = torch.cat((arr, torch.mean(y_raw[:,75:]).unsqueeze(-1) *1e1), dim=-1)
#     return arr

# def domainKnowledgeTile(y_raw):
#     #create a mask to divide the image into central and peripheral regions
#     ind = y_raw.shape[0]
#     mask = np.zeros((ind,ind)).astype(bool)
#     mask[70:-70,70:-70] = True #central region of the image #todo systematic way: CHT?
    
#     arr = torch.Tensor([])
#     arr = torch.cat((arr, torch.mean(y_raw[mask]).unsqueeze(-1) *1e1), dim=-1)
#     arr = torch.cat((arr, torch.mean(y_raw[~mask]).unsqueeze(-1) *1e1), dim=-1)
#     return arr

# #TODO: use thinner one for test case
# #PB changed to tensor
# #TODO: Currently hard coded for the patch size. Need to modify later
# def squareTiles(y_raw, **kwargs):
#     numTiles= kwargs.get('numTiles',1) #default numTiles value
#     arr = torch.Tensor([])
#     pixPerTile = y_raw.shape[0]//numTiles
#     for i in range(numTiles):
#         for j in range(numTiles):
#             tile = torch.mean(y_raw[...,pixPerTile*i:pixPerTile*(i+1),...,pixPerTile*j:pixPerTile*(j+1)]).unsqueeze(-1)
#             arr = torch.cat((arr,tile*1e1),dim=-1)
#     return arr

# def pixelSSE(y_cand_full, y_target_full):
#     err = y_cand_full-y_target_full
#     return err.pow(2).sum().unsqueeze(-1) #np.power(err,2).sum().unsqueeze(-1) 

# def tileSSE(tiles1, tiles2): #SSE of patches (candidate vs. reference)
#     #todo: need error handling for the case when the two tensors have different shapes
#     return (tiles1-tiles2).pow(2).sum().unsqueeze(-1)

# # Loss functions
# def NRMSE(y_pred, y_true):
#     pass

# def SSE():
#     pass

# # Forward 
# def computeY(y_cand_raw,y_ref_raw, method, **kwargs):
#     y_cand_tiles = method(y_cand_raw,**kwargs)
#     y_ref_tiles = method(y_ref_raw,**kwargs)
#     tileTerm = tileSSE(y_cand_tiles,y_ref_tiles)
#     pixelTerm = pixelSSE(y_cand_raw, y_ref_raw)
#     return torch.cat((y_cand_tiles,pixelTerm-tileTerm),dim=-1).unsqueeze(0).to(**tkwargs) #y_pred
