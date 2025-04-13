import torch
from bott.utils import create_circular_mask

class ReductionFunction(torch.nn.Module):

    def __init__(self, reduction_params, device='cuda'):
        super(ReductionFunction, self).__init__()
        self.device = device
        self.reduction_params = reduction_params

    def forward(self, measurement):
        reduction_params = self.reduction_params
        reduction_type   = reduction_params['reduction_type']
        reduction_kwargs = reduction_params['reduction_kwargs']
        
        if reduction_type == 'long':
            arr = get_long_tiles(measurement, **reduction_kwargs)
        elif reduction_type == 'circular':
            arr = get_circular_tiles(measurement, **reduction_kwargs)
        elif reduction_type == 'square':
            arr = get_square_tiles(measurement, **reduction_kwargs)
        else:
            raise ValueError(f"The current implementation does not support reduction type '{reduction_type}'")
        
        return arr

def get_long_tiles(
    measurement: torch.Tensor,
    num_tiles: int = 9,
    tile_width: int = None,
    dim: int = -1,
    reduce = 'mean',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Split the measurement tensor along a specified dimension into tiles.
    Optionally apply reduction (mean) to each tile. Pad the measurement at the end if needed.

    Args:
        measurement: Input tensor of [batch, H, W] or [H, W].
        num_tiles: Number of tiles to generate. If None, calculated from tile_width.
        tile_width: Width of each tile along the given dimension. If None, calculated from num_tiles.
        dim: Dimension along which to split (e.g., -1 = width, -2 = height).
        reduce: If True, return mean of each tile. If False, return full tiles.
        pad_value: Value to pad the measurement if needed.

    Returns:
        A tensor of tile means (if reduce=True) or stacked tiles (if reduce=False).
    """
    total_size = measurement.shape[dim]
    
    # Calculate tile_width based on num_tiles
    if tile_width is None:
        tile_width = total_size // num_tiles
        tile_width = max(1, tile_width)
    
    # Calculate the required size for the measurement to fit exactly num_tiles
    total_required_size = num_tiles * tile_width
    pad_size = max(0, total_required_size - total_size)

    # Pad the measurement at the end if necessary
    if pad_size > 0:
        padding = [0] * (2 * measurement.ndim)
        padding[-(2 * dim + 1)] = pad_size  # Pad at the end of the dimension
        measurement = torch.nn.functional.pad(measurement, padding, value=pad_value)
    # Now we can safely split the measurement into tiles
    tiles = []
    for i in range(num_tiles):
        start = i * tile_width
        end = start + tile_width
        
        slicer = [slice(None)] * measurement.ndim
        slicer[dim] = slice(start, end)
        tile = measurement[tuple(slicer)]
        tiles.append(tile)
    tiles = torch.stack(tiles, dim=-3).to(dtype=measurement.dtype, device=measurement.device)
    
    if reduce == 'mean':
        return tiles.mean(dim=(-2,-1))
    elif reduce == 'sum':
        return tiles.sum(dim=(-2,-1))
    elif reduce in (False, None):
        return tiles
    else:
        raise ValueError(f"The current implementation does not support reduce = '{reduce}', please use either 'mean', 'sum', or 'False'") 

def get_circular_tiles(
    measurement: torch.Tensor,
    radius: float = 0.3,  # Radius as a fraction of min(height, width)
    reduce = 'mean',
) -> torch.Tensor:
    """
    Divide the measurement tensor into two tiles: a circular center region and the surrounding periphery.
    Returns the mean values of these two regions as a tensor.

    Args:
        measurement: Input tensor of [batch, H, W] or [H, W].
        radius: Radius of the circular mask as a fraction of the minimum dimension.

    Returns:
        A tensor with two elements: mean of center region and mean of peripheral region.
    """
    
    # Get the dimensions 
    h_dim = measurement.ndim - 2
    w_dim = measurement.ndim - 1
    
    h_size = measurement.shape[h_dim]
    w_size = measurement.shape[w_dim]
    
    # Create the circular mask
    mask = create_circular_mask(h_size, w_size, radius, device=measurement.device)
    inv_mask = ~mask
    
    # Reshape to flatten all dimensions before spatial dimensions
    flat_shape = (-1, h_size, w_size)  # All dimensions before h_dim combined
    flat_measurement = measurement.reshape(flat_shape)
    
    # Create tensors to store results for each item in the batch
    img_center = []
    img_periphery = []
    
    # For each item in the batch
    for i in range(flat_measurement.shape[0]):
        img = flat_measurement[i]
        
        # Get center region
        img_center.append(img*mask.to(dtype=img.dtype))
        
        # Get peripheral region
        img_periphery.append(img*inv_mask.to(dtype=img.dtype))
    
    # Stack results into a single tensor
    tiles = torch.stack([
        torch.stack(img_center),
        torch.stack(img_periphery)
    ], dim=-3) # tiles = [batch, tiles, H, W]
    
    # Note that the last dimension is the center/periphery dimension
    if reduce == 'mean':
        return tiles.sum(dim=(-2,-1)) / tiles.count_nonzero(dim=(-2,-1))
    elif reduce == 'sum':
        return tiles.sum(dim=(-2,-1))
    elif reduce in (False, None):
        return tiles
    else:
        raise ValueError(f"The current implementation does not support reduce = '{reduce}', please use either 'mean', 'sum', or 'False'") 

def get_square_tiles(
    measurement: torch.Tensor,
    num_tiles: int = 2,
    tile_width: int = None,
    reduce = 'mean',
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Split the measurement tensor into square tiles across the last two dimensions.
    Optionally apply reduction (mean) to each tile. Pad the measurement at the end if needed.

    Args:
        measurement: Input tensor of [batch, H, W] or [H, W].
        num_tiles: Number of tiles in each dimension (creating num_tiles x num_tiles grid).
        tile_width: Width of each tile. If None, calculated from num_tiles.
        reduce: If True, return mean of each tile. If False, return full tiles.
        pad_value: Value to pad the measurement if needed.

    Returns:
        A tensor of tile means (if reduce=True) or stacked tiles (if reduce=False).
    """
    # Get the dimensions for the tiling (last two dimensions)
    h_dim = measurement.ndim - 2
    w_dim = measurement.ndim - 1
    
    h_size = measurement.shape[h_dim]
    w_size = measurement.shape[w_dim]
    
    # Calculate tile width based on num_tiles if not provided
    if tile_width is None:
        h_tile_width = h_size // num_tiles
        w_tile_width = w_size // num_tiles
        h_tile_width = max(1, h_tile_width)
        w_tile_width = max(1, w_tile_width)
    else:
        h_tile_width = tile_width
        w_tile_width = tile_width
    
    # Calculate required sizes and padding
    h_required_size = num_tiles * h_tile_width
    w_required_size = num_tiles * w_tile_width
    
    h_pad_size = max(0, h_required_size - h_size)
    w_pad_size = max(0, w_required_size - w_size)
    
    # Pad the measurement at the end if necessary
    if h_pad_size > 0 or w_pad_size > 0:
        padding = [0] * (2 * measurement.ndim)
        padding[-(2 * h_dim + 1)] = h_pad_size  # Pad at the end of height dimension
        padding[-(2 * w_dim + 1)] = w_pad_size  # Pad at the end of width dimension
        measurement = torch.nn.functional.pad(measurement, padding, value=pad_value)
    
    # Now we can safely split the measurement into tiles
    tiles = []
    for i in range(num_tiles):
        h_start = i * h_tile_width
        h_end = h_start + h_tile_width
        
        for j in range(num_tiles):
            w_start = j * w_tile_width
            w_end = w_start + w_tile_width
            
            slicer = [slice(None)] * measurement.ndim
            slicer[h_dim] = slice(h_start, h_end)
            slicer[w_dim] = slice(w_start, w_end)
            
            tile = measurement[tuple(slicer)]
            tiles.append(tile)
    
    # Stack tiles along a new first dimension
    tiles = torch.stack(tiles, dim=-3).to(dtype=measurement.dtype, device=measurement.device) # [batch, tile_n, reduce_H, reduce_W]
    
    if reduce == 'mean':
        return tiles.mean(dim=(-2,-1))
    elif reduce == 'sum':
        return tiles.sum(dim=(-2,-1))
    elif reduce in (False, None):
        return tiles
    else:
        raise ValueError(f"The current implementation does not support reduce = '{reduce}', please use either 'mean', 'sum', or 'False'") 