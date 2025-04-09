import numpy as np
import torch

class ReductionFunction(torch.nn.Module):

    def __init__(self, reduction_params, device='cuda'):
        super(ReductionFunction, self).__init__()
        self.device = device
        self.reduction_params = reduction_params

    def forward(self, measurement):
        reduction_params = self.reduction_params
        reduction_type   = reduction_params['reduction_type']
        
        if reduction_type == 'vertical':
            arr = get_vertical_tiles(measurement, num_tiles=reduction_params.get('num_tiles'))
        elif reduction_type == 'domain':
            arr = get_circular_tiles(measurement, num_tiles=reduction_params.get('radius'))
        elif reduction_type == 'square':
            arr = get_square_tiles(measurement, num_tiles=reduction_params.get('num_tiles'))
        else:
            raise ValueError(f"The current implementation does not support tiling type {reduction_type}")
        
        return arr

'''
If we want to do GD in the future, we'll need to make sure all these reduciton mehtods are implemented differentiably as well
'''

def get_vertical_tiles(measurement, num_tiles: int=9):
    # TODO For non-overlapping tilts there must be a cleaner way, maybe list comprehension or einops
    arr = torch.Tensor([]).to(torch.double)
    ind = int(measurement.shape[0]//num_tiles)
    for i in range(ind):
        arr = torch.cat((arr, torch.mean(measurement[:,i*ind:(i+1)*ind]).unsqueeze(-1) *1e1), dim=-1)
    return arr

def get_circular_tiles(measurement, radius: float):
    # TODO We need a circular mask with parameterized radius
    #create a mask to divide the image into central and peripheral regions
    ind = measurement.shape[0]
    mask = np.zeros((ind,ind)).astype(bool)
    mask[70:-70,70:-70] = True #central region of the image #todo systematic way: CHT?
    arr = torch.Tensor([])
    arr = torch.cat((arr, torch.mean(measurement[mask]).unsqueeze(-1) *1e1), dim=-1)
    arr = torch.cat((arr, torch.mean(measurement[~mask]).unsqueeze(-1) *1e1), dim=-1)
    return arr

def get_square_tiles(measurement, num_tiles: int=2):
    # TODO For non-overlapping tilts there must be a cleaner way, maybe list comprehension or einops
    arr = torch.Tensor([])
    pixPerTile = int(measurement.shape[0]//num_tiles)
    for i in range(num_tiles):
        for j in range(num_tiles):
            tile = torch.mean(measurement[...,pixPerTile*i:pixPerTile*(i+1),...,pixPerTile*j:pixPerTile*(j+1)]).unsqueeze(-1)
            arr = torch.cat((arr,tile*1e1),dim=-1)
    return arr