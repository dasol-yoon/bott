import torch


class LossFunction(torch.nn.Module):

    def __init__(self, loss_params, device='cuda'):
        super(LossFunction, self).__init__()
        self.device = device
        self.loss_params = loss_params
        
    def forward(self, y_simu, y_true, dp_pow=None, reduce=True):
        loss_params = self.loss_params
        loss_type   = loss_params['loss_type']
        pow      = loss_params.get('dp_pow', 1) if dp_pow is None else dp_pow
        
        if loss_type == 'SSE':
            loss = SSE(y_simu, y_true, pow, reduce=reduce)
        elif loss_type == 'MSE':
            loss = MSE(y_simu, y_true, pow, reduce=reduce)
        elif loss_type == 'NRMSE':
            loss = NRMSE(y_simu, y_true, pow, reduce=reduce)
        else:
            raise ValueError(f"The current implementation does not support lossing type {loss_type}")
        
        return loss
    
'''
If we need some special design of loss (like concatenate different losses), it might be better to define that loss type directly, or to append the loss into a list
'''
        # Forward 
        # def computeY(y_cand_raw,y_ref_raw, method, **kwargs):
        #     y_cand_tiles = method(y_cand_raw,**kwargs)
        #     y_ref_tiles = method(y_ref_raw,**kwargs)
        #     tileTerm = tileSSE(y_cand_tiles,y_ref_tiles)
        #     pixelTerm = pixelSSE(y_cand_raw, y_ref_raw)
        #     return torch.cat((y_cand_tiles,pixelTerm-tileTerm),dim=-1).unsqueeze(0).to(**tkwargs) #y_pred

'''
I'm still unsure what would be the appropriate shape for our loss yet
'''

# Loss functions
def SSE(y_simu, y_true, dp_pow, reduce=True):
    # reduce decides whether we reduce the batch dimension
    y_simu = safe_power(y_simu, dp_pow)
    y_true = safe_power(y_true, dp_pow)
    if reduce:
        # this section is for objective used in EICF
        reduce_dims = [-1] # only sum over n_tiles
        y_true = y_true.unsqueeze(0).unsqueeze(0) #original shape is [1,n_tile] > [1,1,1,n_tiles] to support [num_sample,batch,q,n_tiles]
    else: 
        reduce_dims = tuple(range(-y_true.ndim, 0))
    return (y_simu-y_true).pow(2).sum(dim=reduce_dims)

def MSE(y_simu, y_true, dp_pow, reduce=True):
    y_simu = safe_power(y_simu, dp_pow)
    y_true = safe_power(y_true, dp_pow)
    if reduce:
        reduce_dims = tuple(range(y_simu.ndim))
    else: 
        reduce_dims = tuple(range(1, y_simu.ndim))
    return (y_simu-y_true).pow(2).mean(dim=reduce_dims)

def NRMSE(y_simu, y_true, dp_pow, reduce=True):
    y_simu = safe_power(y_simu, dp_pow)
    y_true = safe_power(y_true, dp_pow)
    data_mean = y_true.mean()
    if reduce:
        reduce_dims = tuple(range(y_simu.ndim))
    else: 
        reduce_dims = tuple(range(1, y_simu.ndim))
    return (y_simu-y_true).pow(2).mean(dim=reduce_dims).sqrt() / data_mean

def safe_power(arr, power, eps=1e-6):
    '''
    Raise the power of measurement in a numerically safe way
    '''
    return (arr+eps).pow(power)