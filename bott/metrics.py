## Put the loss functions and reduction methods
import torch

# Reduction methods
def twoVertTiles(y_raw):
    arr = torch.Tensor([]);
    arr = torch.cat((arr, torch.mean(y_raw[:,:75]).unsqueeze(-1) *1e1), dim=-1)
    arr = torch.cat((arr, torch.mean(y_raw[:,75:]).unsqueeze(-1) *1e1), dim=-1)
    return arr

# Loss functions
def NRMSE(y_pred, y_true):
    pass

def SSE():
    pass