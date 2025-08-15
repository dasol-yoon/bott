"""
Gradient descent module. Many functions are directly imported / modified from PtyRAD https://github.com/chiahao3/ptyrad

"""

import torch
from torch.fft import fft2, ifft2
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur
from bott.utils import get_EM_constants
import ase
import abtem
import numpy as np
import dask
from copy import deepcopy

### Math utils
def fftshift2(x):
    """ A wrapper over torch.fft.fftshift for the last 2 dims """
    # Note that fftshift and ifftshift are only equivalent when N = even 
    return torch.fft.fftshift(x, dim=(-2,-1))  

def ifftshift2(x):
    """ A wrapper over torch.fft.ifftshift for the last 2 dims"""
    # Note that fftshift and ifftshift are only equivalent when N = even 
    return torch.fft.ifftshift(x, dim=(-2,-1))

def scaled_sigmoid(length, offset=0, scale=1, device='cuda'):
    """
    Generate a scaled sigmoid decay over `length` positions, optionally with per-sample offsets.

    Parameters:
        length (int): Number of positions along x-axis.
        offset (float or 1D tensor): Center location(s) where the sigmoid is 0.5.
        scale (float): Controls steepness of decay. Higher means slower drop.

    Returns:
        Tensor of shape (length,) if offset is scalar, or (N, length) if offset is 1D tensor of size N.
    """
    # If scale =  1, y drops from 1 to 0 between (-0.5,0.5), or effectively 1 px
    # If scale = 10, it takes roughly 10 px for y to drop from 1 to 0
    
    x = torch.arange(length, device=device)

    if torch.is_tensor(offset):
        x = x.view(1, -1)  # shape (1, length)
        offset = offset.view(-1, 1)  # shape (N, 1)
    scaled_sigmoid = 1 / (1 + torch.exp((x - offset) / scale * 10))

    return scaled_sigmoid

def torch_phasor(phase):
    """
    Creates a complex tensor with unit magnitude using the phase.

    Args:
        phase (torch.Tensor): phase angle for the exp(i*theta)
        
    Note:
        This util function is created so torch.compile can properly handle complex tensors,
        because torch.exp(1j*phase) involves the 1j which is actually a Python built-in that can't be traced.
    """
    return torch.polar(torch.ones_like(phase), phase)


### Image processing
def meas_flipT(meas, flipT_axes, verbose=True):
    """
    Flip and transpose measurement array.
    flipT_axes: list of 3 binary/int values [flipud, fliplr, transpose]
    """
    if flipT_axes is None:
        return meas

    # Validate length
    if not isinstance(flipT_axes, (list, tuple)) or len(flipT_axes) != 3:
        raise ValueError(f"Expected flipT_axes to be a list of 3 values, got: {flipT_axes}")

    # Safely cast all entries to int
    try:
        flipT_axes = [int(v) for v in flipT_axes]
    except Exception as e:
        raise ValueError(f"flipT_axes must contain values convertible to int (0 or 1). Got: {flipT_axes}") from e
    if verbose:
        print(f"Flipping measurements with [flipud, fliplr, transpose] = {flipT_axes}")

    if flipT_axes[0]:
        meas = torch.flipud(meas)
    if flipT_axes[1]:
        meas = torch.fliplr(meas)
    if flipT_axes[2]:
        meas = torch.permute(meas, (1, 0))

    return meas

def imshift_batch(img, shifts, grid):
    """
    Generates a batch of shifted images from a single input image (..., Ny,Nx) with arbitray leading dimensions.
    
    This function shifts a complex/real-valued input image by applying phase shifts in the Fourier domain,
    achieving subpixel shifts in both x and y directions.

    Inputs:
        img (torch.Tensor): The input image to be shifted. 
                            img could be either a mixed-state complex probe (pmode, Ny, Nx) complex64 tensor, 
                            or a mixed-state pseudo-complex object stack (2,omode,Nz,Ny,Nx) float32 tensor.
        shifts (torch.Tensor): The shifts to be applied to the image. It should be a (Nb,2) tensor and each slice as (shift_y, shift_x).
        grid (torch.Tensor): The k-space grid used for computing the shifts in the Fourier domain. It should be a tensor with shape=(2, Ny, Nx),
                             where Ny and Nx are the height and width of the images, respectively. Note that the grid is normalized so the value spans
                             from 0 to 1

    Outputs:
        shifted_img (torch.Tensor): The batch of shifted images. It has an extra dimension than the input image, i.e., shape=(Nb, ..., Ny, Nx),
                                    where Nb is the number of samples in the input batch.

    Note:
        - The shifts are in unit of pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions, positive is down/right-ward.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
        - tensor[None, ...] would add an extra dimension at 0, so *[None]*ndim means unwrapping a list of ndim None as [None, None, ...]
        - The img is automatically broadcast to (Nb, *img.shape), so if a batch of images are passed in, each image would be shifted independently
    """
    
    assert img.shape[-2:] == grid.shape[-2:], f"Found incompatible dimensions. img.shape[-2:] = {img.shape[-2:]} while grid.shape[-2:] = {grid.shape[-2:]}"
    
    ndim = img.ndim                                                                   # Get the total img ndim so that the shift is dimension-indepent
    shifts = shifts[(...,) + (None,) * ndim]                                          # Expand shifts to (Nb,2,1,1,...) so shifts.ndim = ndim+2. It was written as `shifts = shifts[..., *[None]*ndim]` for Python 3.11 or above with better readability
    grid = grid[(slice(None),) + (None,) * (ndim - 1) + (...,)]                       # Expand grid to (2,1,1,...,Ny,Nx) so grid.ndim = ndim+2. It was written as `grid = grid[:,*[None]*(ndim-1), ...]` for Python 3.11 or above with better readability
    shift_y, shift_x = shifts[:, 0], shifts[:, 1]                                     # shift_y, shift_x are (Nb,1,1,...) with ndim singletons, so the shift_y.ndim = ndim+1
    ky, kx = grid[0], grid[1]                                                         # ky, kx are (1,1,...,Ny,Nx) with ndim-2 singletons, so the ky.ndim = ndim+1
    w = torch.exp(-(2j * torch.pi) * (shift_x * kx + shift_y * ky))                   # w = (Nb, 1,1,...,Ny,Nx) so w.ndim = ndim+1. w is at the center.
    shifted_img = ifft2(ifftshift2(fftshift2(fft2(img)) * w))                         # For real-valued input, take shifted_img.real
    
    # Note that for imshift, it's better to keep fft2(img) than fft2(ifftshift2(img))
    # While fft2(img).angle() might seem serrated, it's indeed better to keep it as is, which is essentially setting the center as the origin for FFT.
    
    return shifted_img


### Physics
def near_field_evolution_torch(Npix_shape, dx, dz, lambd, device='cuda'):
    """Fresnel propagator in PyTorch"""
    # This is for testing and demonstration purpose and is never called in PtyRAD
    # The actual optimizable Fresnel propagator is directly built inside "PtychoAD.get_propagators"
    dx    = dx.to(device)
    dz    = dz.to(device)
    lambd = lambd.to(device)

    ygrid = (torch.arange(-Npix_shape[-2] // 2, Npix_shape[-2] // 2, device=device) + 0.5) / Npix_shape[-2]
    xgrid = (torch.arange(-Npix_shape[-1] // 2, Npix_shape[-1] // 2, device=device) + 0.5) / Npix_shape[-1]

    # Standard ASM
    k  = 2 * torch.pi / lambd
    ky = 2 * torch.pi * ygrid / dx
    kx = 2 * torch.pi * xgrid / dx
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
    H = torch.fft.ifftshift(torch.exp(1j * dz * torch.sqrt(k ** 2 - Kx ** 2 - Ky ** 2))) # H has zero frequency at the corner in k-space

    return H

def make_stem_probe(params_dict, verbose=True):
    # MAKE_TEM_PROBE Generate probe functions produced by object lens in 
    # transmission electron microscope.
    # Written by Yi Jiang based on Eq.(2.10) in Advanced Computing in Electron 
    # Microscopy (2nd edition) by Dr.Kirkland
    # Implemented and slightly modified in python by Chia-Hao Lee
 
    # Outputs:
        #  probe: complex probe functions at real space (sample plane)
    # Inputs: 
        #  params_dict: probe parameters and other settings
    
    import numpy as np
    from numpy.fft import fftfreq, fftshift, ifft2, ifftshift
    
    ## Basic params
    voltage     = float(params_dict["kv"])         # Ang
    conv_angle  = float(params_dict["conv_angle"]) # mrad
    Npix        = int  (params_dict["Npix"])       # Number of pixel of thr detector/probe
    dx          = float(params_dict["dx"])         # px size in Angstrom
    ## Aberration coefficients
    df          = float(params_dict["df"]) #first-order aberration (defocus) in angstrom
    c3          = float(params_dict["c3"]) #third-order spherical aberration in angstrom
    c5          = float(params_dict["c5"]) #fifth-order spherical aberration in angstrom
    c7          = float(params_dict["c7"]) #seventh-order spherical aberration in angstrom
    f_a2        = float(params_dict["f_a2"]) #twofold astigmatism in angstrom
    f_a3        = float(params_dict["f_a3"]) #threefold astigmatism in angstrom
    f_c3        = float(params_dict["f_c3"]) #coma in angstrom
    theta_a2    = float(params_dict["theta_a2"]) #azimuthal orientation in radian
    theta_a3    = float(params_dict["theta_a3"]) #azimuthal orientation in radian
    theta_c3    = float(params_dict["theta_c3"]) #azimuthal orientation in radian
    shifts      = params_dict["shifts"] #shift probe center in angstrom
    
    # Calculate some variables
    wavelength = 12.398/np.sqrt((2*511.0+voltage)*voltage) #angstrom
    k_cutoff = conv_angle/1e3/wavelength
    dk = 1/(dx*Npix)
    
    if verbose:
        print("Start simulating STEM probe")
    
    # Make k space sampling and probe forming aperture
    kx = fftshift(fftfreq(Npix, 1/Npix))
    # kx = np.linspace(-np.floor(Npix/2),np.ceil(Npix/2)-1,Npix)
    kX,kY = np.meshgrid(kx,kx, indexing='xy')

    kX = kX*dk
    kY = kY*dk
    kR = np.sqrt(kX**2+kY**2)
    theta = np.arctan2(kY,kX)
    mask = (kR<=k_cutoff).astype('bool') 
    
    # Adding aberration one-by-one, the aberrations modify the flat phase (imagine a flat wavefront at aperture plane) with some polynomial perturbations
    # The aberrated phase is called chi(k), probe forming aperture is placed here to select the relatively flat phase region to form desired real space probe
    # Note that chi(k) is real-valued function with unit as radian, it's also not limited between -pi,pi. Think of phase shift as time delay might help.
    
    chi = -np.pi*wavelength*kR**2*df
    if c3!=0: 
        chi += np.pi/2*c3*wavelength**3*kR**4
    if c5!=0: 
        chi += np.pi/3*c5*wavelength**5*kR**6
    if c7!=0: 
        chi += np.pi/4*c7*wavelength**7*kR**8
    if f_a2!=0: 
        chi += np.pi*f_a2*wavelength*kR**2*np.sin(2*(theta-theta_a2))
    if f_a3!=0: 
        chi += 2*np.pi/3*f_a3*wavelength**2*kR**3*np.sin(3*(theta-theta_a3))
    if f_c3!=0: 
        chi += 2*np.pi/3*f_c3*wavelength**2*kR**3*np.sin(theta-theta_c3)

    psi = np.exp(-1j*chi)*np.exp(-2*np.pi*1j*shifts[0]*kX)*np.exp(-2*np.pi*1j*shifts[1]*kY)
    probe = mask*psi # It's now the masked wave function at the aperture plane
    probe = fftshift(ifft2(ifftshift(probe))) # Propagate the wave function from aperture to the sample plane. 
    probe = probe/np.sqrt(np.sum((np.abs(probe))**2)) # Normalize the probe so sum(abs(probe)^2) = 1

    if verbose:
        # Print some useful values
        print(f'kv          = {voltage} kV')    
        print(f'wavelength  = {wavelength:.4f} Ang')
        print(f'conv_angle  = {conv_angle} mrad')
        print(f'Npix        = {Npix} px')
        print(f'dk          = {dk:.4f} Ang^-1')
        print(f'kMax        = {(Npix*dk/2):.4f} Ang^-1')
        print(f'alpha_max   = {(Npix*dk/2*wavelength*1000):.4f} mrad')
        print(f'dx          = {dx:.4f} Ang, Nyquist-limited dmin = 2*dx = {2*dx:.4f} Ang')
        print(f'Rayleigh-limited resolution  = {(0.61*wavelength/conv_angle*1e3):.4f} Ang (0.61*lambda/alpha for focused probe )')
        print(f'Real space probe extent = {dx*Npix:.4f} Ang')
    
    return probe

def multislice_forward_model_vec_all(probe, H, tmask, objp_patches, eps=1e-10):
    """
    Computes the multislice position-averaged electron diffraction pattern (PACBED) 
    with AD-optimizable thickness and tilts using probe and mixed-state object
    using a fully vectorized forward model.

    Args:
        probe (torch.Tensor): Tensor of shape (N, Ny, Nx) with complex128 values, 
            representing the probe(s). N is the number of samples in the batch.
        H (torch.Tensor): Tensor of shape (num_tilts, Ky, Kx) with complex128 values, representing the Fresnel 
            propagator that propagates the wave by a slice thickness given a tilt.
        tmask (torch.Tensor): Tensor of shape (num_thickness, Nz) with float64 values, representing
            the sigmoid mask used to zero out the contribution from object slices beyond the target slice (offset) 
        objp_patches (torch.Tensor): Tensor of shape (N, omode, Nz, Ny, Nx), representing
            object phase patches with float32. 
            N is the number of samples in a batch, omode is the number of object modes used for FrozenPhonons, 
            Nz, Ny, Nx are the dimensions of the object patches.
        eps (float, optional): A small value added for numerical stability. Defaults to 1e-10.

    Returns:
        torch.Tensor: Tensor of shape (N, num_tilts, num_thickness, Ky, Kx) with float64 positive values, representing the 
        forward diffraction patterns for each tilt and thickness.
    """
    
    # Multiply object phase with the thickness mask
    objp_patches = objp_patches[:,None] * tmask[...,None,:,None,None] # Final shape of objc_patches (N, num_thickness, omode, Nz, Ny, Nx)
    
    # Cast the object back to actual complex tensor
    object_cplx = torch_phasor(objp_patches)
    n_slices = object_cplx.shape[-3]

    # Expand psi to include num_tilts, num_thickness, and omode dimension
    psi = probe[:, None, None, None, :, :] # (N, Ny, Nx) -> (N, 1, 1, 1, Ny, Nx) -> (N, num_tilts, num_thickness, omode, Ny, Nx)

    # Propagating each object layer using broadcasting
    for n in range(n_slices-1):
        object_slice = object_cplx[..., n, :, :]  # object_slice -> (N, num_thickness, omode, Ny, Nx)
        psi = psi * object_slice[:, None] # psi -> (N, num_tilts, num_thickness, omode, Ny, Nx). Note that psi is always centered in real space
        psi = ifft2(H[:,None,None] * fft2(psi))    # H -> (num_tilts, 1, 1, Ny, Nx). Note that fft2 and ifft2 are applying to the last 2 axes. 

    # Interacting with the last layer, and no propagation is needed afterward
    object_slice = object_cplx[..., n_slices-1, :, :]
    psi = psi * object_slice[:, None]

    # Propagate the object-modified exit wave psi(r) to detector plane into psi(k)
    # Averaged over the omode dimension
    # dp_fwd (N, num_tilts, num_thickness, Ny, Nx)
    dp_fwd = (
        torch.mean(
            (fftshift2(fft2(psi, norm="ortho"))).abs().square(),
            dim=-3,
        )
        + eps
    )
    return dp_fwd


### Model utils
def toggle_grad_requires(model, niter, verbose=True):
    """Toggle requires_grad based on start and end iteration for each optimizable tensor."""
    if verbose:
        print(" ") # Empty line for the start of each iteration
    
    optimizable_tensors = model.optimizable_tensors
    for param_name in model.optimizable_tensors.keys():
        start_iter = model.start_iter.get(param_name)
        end_iter = model.end_iter.get(param_name)
        
        # Determine if gradients should be enabled
        grad_started = start_iter is not None and niter >= start_iter
        grad_ended = end_iter is not None and niter + 1 > end_iter # end_iter is exclusive
        requires_grad = grad_started and not grad_ended
        
        optimizable_tensors[param_name].requires_grad = requires_grad
        if verbose:
            print(f"Iter: {niter}, {param_name}.requires_grad = {requires_grad}")

def create_optimizer(optimizer_params, optimizable_params):
    # Extract the optimizer name and configs
    optimizer_name = optimizer_params['name']
    optimizer_configs = optimizer_params.get('configs') or {} # if "None" is provided or missing, it'll default an empty dict {}
    
    print(f"### Creating PyTorch '{optimizer_name}' optimizer with configs = {optimizer_configs} ###")
    
    # Get the optimizer class from torch.optim
    optimizer_class = getattr(torch.optim, optimizer_name, None)
    
    if optimizer_class is None or optimizer_name == 'LBFGS':
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")

    optimizer = optimizer_class(optimizable_params, **optimizer_configs)
    return optimizer

### Visualization
def plot_cbed_difference(dp_fwd, dp_meas, dp_pow=0.5):
    """ Visualization function that plots the CBED with optional I^power """
    import matplotlib.pyplot as plt
    
    dp_fwd = dp_fwd**dp_pow
    dp_meas = dp_meas**dp_pow
    diff = dp_fwd - dp_meas

    fig, axs = plt.subplots(1,3, figsize=(16,5))
    plt.suptitle(f"DP pow = {dp_pow}", y=0.90)
    im0 = axs[0].imshow(dp_fwd)
    im1 = axs[1].imshow(dp_meas)
    im2 = axs[2].imshow(diff, cmap='seismic', vmin=-3*diff.std(), vmax=3*diff.std())

    axs[0].set_title(f'Forward PACBED')
    axs[1].set_title('Measured PACBED')
    axs[2].set_title('Difference (forward - measured)')

    fig.colorbar(im0, shrink=0.6)
    fig.colorbar(im1, shrink=0.6)
    fig.colorbar(im2, shrink=0.6)
    plt.show()

### Major optimization class
class GDTT(torch.nn.Module):
    def __init__(self, model_params, device='cuda', verbose=True):
        super(GDTT, self).__init__()
        with torch.no_grad():
            
            print("### Initializing GDTT model ###")
            
            # Setup model params
            self.model_params           = model_params
            self.device                 = device
            self.verbose                = verbose
            self.meas_flipT             = model_params['meas_flipT']
            self.model_probe_params     = model_params['probe_params']
            self.detector_blur_std      = model_params['detector_blur_std']
            self.return_pacbed          = model_params['return_pacbed']
            
            # Parse the learning rate and start iter for optimizable tensors
            start_iter_dict = {}
            end_iter_dict = {}
            lr_dict = {}
            for key, params in model_params['update_params'].items():
                start_iter_dict[key] = params.get('start_iter')
                end_iter_dict[key] = params.get('end_iter')
                lr_dict[key] = params['lr']
            self.optimizer_params       = model_params['optimizer_params']
            self.start_iter             = start_iter_dict
            self.end_iter               = end_iter_dict
            self.lr_params              = lr_dict
            
            # Scaling factors for numerical stability and optimization efficiency
            self.scale_dp               = model_params['scale_dp']
            self.scale_tilt             = model_params['scale_tilt']
            self.scale_thickness        = model_params['scale_thickness']

            # Optimizable tensors, use float64 for better numerical stability
            self.opt_tilts              = nn.Parameter(torch.tensor(model_params['init_tilts'] / self.scale_tilt, device=device, dtype=torch.float64))
            self.opt_thicknesses        = nn.Parameter(torch.tensor(model_params['init_thicknesses'] / self.scale_thickness, device=device, dtype=torch.float64))
            
            # Initialize
            self.dx                     = torch.tensor(model_params['dx'],                                         device=device)
            self.dz                     = torch.tensor(model_params['dz'],                                         device=device)
            self.lambd                  = torch.tensor(get_EM_constants(self.model_probe_params['kv'], 'wavelength'), device=device)
            self.raw_meas               = torch.tensor(model_params['raw_meas'], dtype=torch.float64, device=device)
            self.init_potential         = torch.tensor(model_params['init_potential'], dtype=torch.float64, device=device) # (omode,z,y,x)
            self.init_pos_ang_yx        = torch.tensor(model_params['init_pos_ang_yx'], dtype=torch.float64, device=device)
            self.dp_meas                = self.init_dp_meas()
            self.probe                  = self.init_probe()
            self.objp                   = self.init_objp()
            self.init_grids()
            self.init_pos()
            self.H                      = near_field_evolution_torch(self.probe.shape[-2:], self.dx, self.dz, self.lambd)
            self.loss_iters             = []
            self.tilts_iters            = []
            self.thicknesses_iters      = []
            
            # Create a dictionary to store the optimizable tensors
            self.optimizable_tensors = {
                'tilts'       : self.opt_tilts,
                'thicknesses' : self.opt_thicknesses}
            self.create_optimizable_params_dict(self.lr_params)
            
            self.print_model_summary()
            
            print("### Done initializing GDTT model ###")
    
    def create_optimizable_params_dict(self, lr_params):
        """ Sets the optimizer with lr_params """
        # # Use this to edit learning rate if needed some refinement

        # model.set_optimizer(lr_params={'obja'            : 5e-4,
        #                                'objp'            : 5e-4,
        #                                'obj_tilts'       : 1e-4,
        #                                'probe'           : 1e-4, 
        #                                'probe_pos_shifts': 1e-4})
        # optimizer=torch.optim.Adam(model.optimizer_params)
        
        self.lr_params = lr_params
        self.optimizable_params = []
        for param_name, lr in lr_params.items():
            if param_name not in self.optimizable_tensors:
                raise ValueError(f"WARNING: '{param_name}' is not a valid parameter name, check your `update_params` and choose from 'tilts', 'thicknesses'")
            else:
                self.optimizable_tensors[param_name].requires_grad = (lr != 0) and (self.start_iter[param_name] ==1) # Set requires_grad based on learning rate and start_iter
                if lr != 0:
                    self.optimizable_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})               
    
    def print_model_summary(self):
        pass
    
    def init_dp_meas(self):
        """ Initialize raw measurement (i.e., resampling, scaling) """
        Npix = self.model_probe_params['Npix']
        
        # Resampling
        scale_factors = [Npix/self.raw_meas.shape[-2], Npix/self.raw_meas.shape[-1]]
        meas = torch.nn.functional.interpolate(self.raw_meas[None, None,], scale_factor=scale_factors, mode='bilinear').squeeze() # 2D interpolate requires 4D input (N, C, H, W) #/ math.prod(scale_factors) # Dividing by scale_factors would keep meas.sum() roughly conserved
        
        # Normalize it such that the CBED sum to 1
        meas /= meas.sum() 
        
        # Flipping / Transpose
        if self.meas_flipT is not None:
            meas = meas_flipT(meas, flipT_axes=self.meas_flipT, verbose=self.verbose) # Apply transpose to match the orientation between abTEM and PtyRAD, note that this transpose may not be needed if the input meas is experimental data
        return meas
            
    def init_probe(self):
        model_probe_params = self.model_probe_params
        probe = torch.tensor(make_stem_probe(model_probe_params, verbose=self.verbose), dtype=torch.complex128, device=self.device) # The probe is by default intensity sum at 1
        return probe
        
    def init_objp(self):
        # Initialize object
        
        potential = self.init_potential
        kv = self.model_probe_params['kv']
        
        scale_factors = [2/3, 2/3] # Resample the pixel size to 3/2 because the kMax was crop to 2/3 by abTEM. Presumably we can resample z to get more compute efficiency with reduced accuracy
        potential_resample = torch.nn.functional.interpolate(potential, scale_factor=scale_factors, mode='bilinear') # The potential value is now resampled but no need to scale the value.
        objp = get_EM_constants(kv, 'sigma') * potential_resample/1e3 # Convert potential to phase shift for the complex object
        return objp
    
    def init_grids(self):
        """ Create the grid for obj_ROI and shift_probes in a vectorized approach """
        Npy, Npx = self.probe.shape[-2:]

       # Grids for Fresnel propagator
       # Note that this grid has a half-bin shift that avoids exact 0 frequency which would generate NaNs inside sqrt.
        ygrid = (torch.arange(-Npy // 2, Npy // 2, device=self.device) + 0.5) / Npy
        xgrid = (torch.arange(-Npx // 2, Npx // 2, device=self.device) + 0.5) / Npx
        ky = torch.fft.ifftshift(2 * torch.pi * ygrid / self.dx) # Use ifftshift to shift the 0 frequency to the corner
        kx = torch.fft.ifftshift(2 * torch.pi * xgrid / self.dx)
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
        self.propagator_grid = torch.stack([Ky,Kx], dim=0) # (2,Ky,Kx), k-space grid for Fresnel propagator with 2pi absorbed

        # Grids for shifting probes and obj_ROI selection
        rpy_grid, rpx_grid = torch.meshgrid(torch.arange(Npy, dtype=torch.int32, device=self.device), 
                                            torch.arange(Npx, dtype=torch.int32, device=self.device), indexing='ij') # real space grid for probe in y and x directions
        self.rpy_grid = rpy_grid
        self.rpx_grid = rpx_grid
        self.shift_probes_grid = torch.stack([rpy_grid/Npy, rpx_grid/Npx], dim=0) # (2,Npy,Npx), normalized k-space grid stack for sub-px probe shifting 
    
    def init_pos(self):
        # Preprocess the position so that it's compatible with follow up reconstruction packages
        pos_ang_yx = self.init_pos_ang_yx
        dx = self.dx
        probe_shape = torch.tensor(self.probe.shape[-2:], device=self.device)
        
        pos_px_yx = pos_ang_yx / dx
        pos_px_yx = pos_px_yx - probe_shape / 2 # Shift back to obj coordinate 

        self.crop_pos         = (torch.round(pos_px_yx)).to(dtype=torch.int32, device=self.device)
        self.probe_pos_shifts = (pos_px_yx - torch.round(pos_px_yx)).to(dtype=torch.float64, device=self.device)
    
    def get_dp_meas(self):
        return self.scale_dp * self.dp_meas
    
    def get_tilts(self):
        return self.scale_tilt * self.opt_tilts
    
    def get_thicknesses(self):
        return self.scale_thickness * self.opt_thicknesses
    
    def get_probes(self, indices):
        """ Get probes for each position """
        probes = imshift_batch(self.probe, shifts=self.probe_pos_shifts[indices], grid=self.shift_probes_grid) # (N, Ny, Nx)
        return probes
    
    def get_propagators(self):
        dz          = self.dz
        Ky, Kx      = self.propagator_grid
        tilts       = self.get_tilts()
        tilts_y     = tilts[:,0,None,None] / 1e3 # mrad, tilts_y = (N,Y,X)
        tilts_x     = tilts[:,1,None,None] / 1e3
        propagators = self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
        return propagators
    
    def get_tmask(self, eps=1e-8):
        length = self.objp.shape[-3] # Number of object z-slices
        offset = self.get_thicknesses() / self.dz
        tmask = scaled_sigmoid(length=length, offset=offset, scale=1) + eps
        return tmask
    
    def get_objp_patches(self, indices):
        """
        Get object patches from specified indices
        
        """
        # Get object patches
        obj_ROI_grid_y = self.rpy_grid[None,:,:] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None,:,:] + self.crop_pos[indices, None, None, 1]
        objp_patches = self.objp[:,:,obj_ROI_grid_y,obj_ROI_grid_x].permute(2,0,1,3,4)
        
        return objp_patches
    
    def get_forward_meas(self, probes, propagators, tmask, objp_patches, eps=1e-10):
        
        dp_fwd = multislice_forward_model_vec_all(probes, propagators, tmask, objp_patches, eps=eps)
        
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)
        
        if self.return_pacbed:
           dp_fwd = dp_fwd.mean(0)
        
        return self.scale_dp * dp_fwd
    
    def forward(self, indices):
        """ Doing the forward pass and get an output diffraction pattern for each input index """
        probes         = self.get_probes(indices)
        propagators    = self.get_propagators()
        tmask          = self.get_tmask()
        objp_patches   = self.get_objp_patches(indices)
        dp_fwd         = self.get_forward_meas(probes, propagators, tmask, objp_patches)
        return dp_fwd
        

### Temporary abTEM class, should merge it with BOTT's physics_models.py module at some point
class SimuAbTEM:
    def __init__(self, params_abtem, device='gpu', verbose=True):
        self.params_abtem = deepcopy(params_abtem)
        self.device = device
        self.verbose = verbose
        
        # Setup device
        if device == 'gpu':
            import cupy as xp
            abtem.config.set({"dask.chunk-size-gpu" : "2048 MB"})
            dask.config.set({"num_workers": 1})
        elif device == 'cpu':
            import numpy as xp
        else:
            raise ValueError(f"device '{device}' not implemented yet, please use 'cpu', or 'gpu'!")
        
        # abtem configure
        abtem.config.set({"local_diagnostics.progress_bar": False})
        abtem.config.set({"device": device})
        self.xp = xp
        
        # Initialization
        self.parse_params(params_abtem)
        self.simulate_cell()
        self.simulate_potential()
        self.simulate_probe()
        self.simulate_pos()
    
    def parse_params(self, params_abtem):
        for key, value in params_abtem.items():
            setattr(self, key, value)
    
    def simulate_cell(self):
        
        # Parse params        
        path_crystal = self.path_crystal
        potential_extent_x = self.potential_extent_x
        potential_extent_y = self.potential_extent_y
        thickness = self.thickness
        
        print("### Simulate cell ###")        
        unit_cell = ase.io.read(path_crystal)
        target_object_extent = np.array((potential_extent_x, potential_extent_y, thickness)) # Specimen range in Ang (x,y,z)
        cell_constants = np.diag(unit_cell.cell)
        super_cell_reps = np.ceil(target_object_extent / cell_constants).astype('int')
        super_cell = unit_cell * super_cell_reps
        print(f"super_cell = {super_cell}")
        
        # Saving attributes
        self.cell_constants = cell_constants
        self.super_cell = super_cell
        
        return super_cell
        
    def simulate_potential(self):
        
        # Parse params
        use_frozen_phonon = self.use_frozen_phonon
        num_phonon_configs = self.num_phonon_configs
        phonon_sigma = self.phonon_sigma
        random_seed = self.random_seed
        super_cell = self.super_cell
        lateral_sampling = self.lateral_sampling
        vertical_sampling = self.vertical_sampling
        potential_parametrization = self.potential_parametrization
        potential_projection = self.potential_projection
        exit_planes = self.exit_planes
        
        print("### Simulate potential ###")        
        if use_frozen_phonon:
            print(f"Using FrozenPhonons potential with {num_phonon_configs} configs")
            atoms = abtem.FrozenPhonons(atoms=super_cell, num_configs=num_phonon_configs, sigmas=phonon_sigma, seed=random_seed)
        else:
            print("Using Static potential")
            atoms = super_cell

        potential = abtem.Potential(atoms=atoms, sampling=lateral_sampling, parametrization=potential_parametrization,
                slice_thickness=vertical_sampling, projection=potential_projection, exit_planes=exit_planes)

        potential_arr = potential.build().compute(progress_bar=False).array

        if potential_arr.ndim == 3:
            potential_arr = potential_arr[None,] # Normalize the potential dimension to 4D with omode (FrozenPhonon) dimension.
        potential_arr = potential_arr.transpose(0,1,3,2)
        print("Note that the last 2 axes are transposed because abTEM go with (omode,z,x,y) but we want (omode,z,y,x)")

        if self.device == 'gpu':
            potential_arr = potential_arr.get()
            
         # Saving attributes
        self.potential = potential
        self.potential_arr = potential_arr

        if self.verbose:
            print(f"potential.shape = {potential.shape}")
            print(f"potential_arr.shape = {potential_arr.shape}")
            
        return potential
        
    def simulate_probe(self):
        # Calculate the probe
        
        energy = self.energy
        convergence_angle = self.convergence_angle
        df = self.df
        tilt_x = self.tilt_x
        tilt_y = self.tilt_y
        aberrations = self.aberrations
        potential = self.potential
        lateral_sampling = self.lateral_sampling
        
        probe = abtem.Probe(energy=energy, semiangle_cutoff=convergence_angle, defocus=df, tilt=(tilt_x, tilt_y), **aberrations)
        probe.grid.match(potential)
        probe_arr = probe.build().compute().array

        if self.device == 'gpu':
            probe_arr = probe_arr.get()

        # Useful information
        wavelength = get_EM_constants(energy/1e3, 'wavelength')
        kmax_antialias = 1/lateral_sampling/3 # 1/Ang #The kmax_antialiasing = 2.675 Ang-1 
        alpha_max_antialias = wavelength * kmax_antialias # rad
        
        # Saving attributes
        self.probe = probe
        
        if self.verbose:
            print(f"Energy = {energy/1e3} kV, rel. wavelength = {wavelength:.4g} Ang")
            print(f"CBED collection kmax = {kmax_antialias:.4g} 1/Ang, collection alpha_max = {alpha_max_antialias*1000:.4g} mrad")
            print(f"probe.shape = {probe.shape}")
            print(f"probe.axes_metadata = {probe.axes_metadata}")
            
        return probe
        
    def simulate_pos(self):
        # Make scan positions, unit in Ang

        # Parse params
        cell_constants = self.cell_constants
        scan_step_size = self.scan_step_size
        probe = self.probe
        
        N_scan_fast = np.ceil(cell_constants[0] / scan_step_size).astype(int)
        N_scan_slow = np.ceil(cell_constants[1] / scan_step_size).astype(int)

        pos = scan_step_size * np.array([(y, x) for y in range(N_scan_slow) for x in range(N_scan_fast)]) # (N,2), each row is (y,x)
        pos = pos - pos.mean(0) # Center scan around origin

        # Apply offset to move the scan pattern to the center of the supercell
        offset = probe.extent[0]/2
        pos += offset

        # Parse the position into the hdf5 for reconstuction, and the abTEM scan position
        pos_ang_yx = pos
        pos_ang_xy = np.flip(pos,1)

        # Saving attributes
        self.pos_ang_yx = pos_ang_yx
        self.pos_ang_xy = pos_ang_xy
        
        if self.verbose:
            print(f"N_scan_fast = {N_scan_fast}")
            print(f"N_scan_slow = {N_scan_slow}")
            print(f"pos_ang_xy[:5] = {pos_ang_xy[:5]}")
            
        return pos_ang_xy
        
    def simulate_cbed(self):
        # Get cbeds, the output would be either (kx, ky) or (nscan, kx, ky) because the phonon dimension would be reduced
        
        probe = self.probe
        pos_ang_xy = self.pos_ang_xy
        potential = self.potential
        return_pacbed = self.return_pacbed
        xp = self.xp
        
        cbeds = probe.multislice(scan = pos_ang_xy, potential = potential).diffraction_patterns(max_angle='cutoff').reduce_ensemble().compute().array

        if return_pacbed and cbeds.ndim == 3:
            measurement_arr = xp.mean(cbeds, axis=(0)) # Reduce the scan axes
        else:
            measurement_arr = cbeds
            
        if self.device == 'gpu':
            measurement_arr = measurement_arr.get()
            
        return measurement_arr