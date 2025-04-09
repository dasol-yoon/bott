# utils are for helper functions that don't fit elsewhere
import numpy as np
import torch

def create_circular_mask(
    height: int,
    width: int,
    radius: float = 0.3,  # Radius as a fraction of min(height, width)
    center: tuple = None,
    device = None
) -> torch.Tensor:
    """
    Create a circular binary mask.

    Args:
        height: Height of the mask.
        width: Width of the mask.
        radius: Radius of the circle as a fraction of the minimum dimension.
        center: Center coordinates (y, x) of the circle. If None, uses the center of the image.
        device: Device to create the mask on. If None, uses default device.

    Returns:
        A binary mask tensor where True indicates pixels inside the circle.
    """
    if center is None:
        center = (height / 2, width / 2)
    
    # Create a meshgrid for the mask
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    
    # Calculate distance from center for each pixel
    distance = torch.sqrt((y - center[0])**2 + (x - center[1])**2)
    
    # Create the circular mask
    r = min(height, width) * radius
    mask = distance <= r
    
    return mask

def make_output_filenm(X):
    X = X.tolist()
    thickness = str(round(X[0])).zfill(4)
    tilt_x = str(round(X[1]))
    tilt_y = str(round(X[2]))
    return f"Thickness_{thickness}_TiltX_{tilt_x}_TiltY_{tilt_y}.tif"

def normalize_arr(arr, norm_type='zero_to_one'):
    
    if norm_type == 'zero_to_one':
        norm_arr = (arr - arr.min()) / (arr.max() - arr.min())
    
    elif norm_type == 'standardize':
        norm_arr = (arr - arr.mean()) / arr.std()
    
    else:
        raise ValueError(f"norm_type '{norm_type}' not implemented yet, please use 'zero_to_one', or 'standardize'!")
    
    return norm_arr

def get_default_abTEM_params():
    params_abTEM = {  
        # Device configuration
        "device_abtem": "gpu",

        # Crystal structure input
        "path_crystal": "./data/SrTiO3.cif",

        # Potential parameters
        "potential_extent_x": 62.6,  # Angstrom
        "potential_extent_y": 62.6,  # Angstrom
        "lateral_sampling": 0.2 * 2 / 3,  # Angstrom
        "vertical_sampling": 2,
        "potential_parametrization": "lobato",  # or "kirkland"
        "potential_projection": "finite",       # or "infinite"

        # Phonon parameters
        "random_seed": 42,
        "use_frozen_phonon": False,
        "num_phonon_configs": 5,
        "phonon_sigma": {
            'Sr': 0.088,
            'Ti': 0.0746,
            'O': 0.0963,
        },

        # Probe parameters
        "energy": 200e3,  # in eV
        "convergence_angle": 19.1,  # mrad
        "df": 0,
        "aberrations": {},

        # Scan parameters
        "scan_step_size": 0.3,  # nm
        "return_pacbed": True,
    }
   
    return params_abTEM

def potential_to_phase(projected_atomic_potential, acceleration_voltage):
    
    # proj_potential: V-Ang
    # acceleration_voltage: kV
    
    sigma = get_EM_constants(acceleration_voltage, 'sigma')
    
    phase_shift = np.angle(np.exp(1j*sigma * projected_atomic_potential/1E3)) # radian in strong phase approximation
    
    return  phase_shift

def get_EM_constants(acceleration_voltage, output_type):
    
    # acceleration_voltage: kV
    
    # Physical Constants
    PLANCKS = 6.62607015E-34 # m^2*kg / s
    REST_MASS_E = 9.1093837015E-31 # kg
    CHARGE_E = 1.602176634E-19 # coulomb 
    SPEED_OF_LIGHT = 299792458 # m/s
    
    # Useful constants in EM unit 
    hc = PLANCKS * SPEED_OF_LIGHT / CHARGE_E*1E-3*1E10 # 12.398 keV-Ang, h*c
    REST_ENERGY_E = REST_MASS_E*SPEED_OF_LIGHT**2/CHARGE_E*1E-3 # 511 keV, m0c^2
    
    # Derived values
    gamma = 1 + acceleration_voltage / REST_ENERGY_E # m/m0 = 1 + e*V/m0c^2, dimensionless, Lorentz factor
    wavelength = hc/np.sqrt((2*REST_ENERGY_E + acceleration_voltage)*acceleration_voltage) # Angstrom, lambda = hc/sqrt((2*m0c^2 + e*V)*e*V))
    sigma = 2*np.pi*gamma*REST_MASS_E*CHARGE_E*wavelength/PLANCKS**2 * 1E-20 * 1E3 # interaction parameter, 2 pi*gamma*m0*e*lambda/h^2, 1/kV-Ang
    
    if output_type == 'gamma':
        return gamma
    elif output_type == 'wavelength':
        return wavelength
    elif output_type == 'sigma':
        return sigma
    else:
        KeyError(f"output_type '{output_type}' not implemented yet, please use 'gamma', 'wavelength', or 'sigma'!")