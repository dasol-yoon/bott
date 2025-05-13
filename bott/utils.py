# utils are for helper functions that don't fit elsewhere
import numpy as np
import torch
from time import perf_counter

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

def get_default_abtem_params():
    params_abtem = {  

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
        "detector_angle":'cutoff',

        # Scan parameters
        "scan_step_size": 0.3,  # nm
        "return_pacbed": True,
    }
   
    return params_abtem

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
        
def print_system_info():
    
    import os
    import platform
    import sys
    import numpy as np
    import torch
    
    print("### System information ###")
    
    # Operating system information
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"OS Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # CPU cores
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        cpus =  int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        # Fallback to the total number of CPU cores on the node
        cpus = os.cpu_count()
    print(f"Available CPU cores: {cpus}")
    
    # Memory information
    if 'SLURM_MEM_PER_NODE' in os.environ:
        # Memory allocated per node by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_NODE']) / 1024  # Convert MB to GB
        print(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    elif 'SLURM_MEM_PER_CPU' in os.environ:
        # Memory allocated per CPU by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_CPU']) * cpus / 1024  # Convert MB to GB
        print(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    else:
        try:
            import psutil
            # Fallback to system memory information
            mem = psutil.virtual_memory()
            print(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
            print(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
        except ImportError:
            print("Memory information will be available after `conda install conda-forge::psutil`")
    
    # GPU information
    if torch.backends.cuda.is_built() and torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available CUDA GPUs: {[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]}")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        print(f"MPS Available: {torch.backends.mps.is_available()}")
    elif torch.backends.cuda.is_built() or torch.backends.mps.is_built():
        print("GPU support built with PyTorch, but could not find any GPU device.")
    else:
        print("No GPU backend (CUDA or MPS) built into this PyTorch install.")
        print("Install a PyTorch version with GPU support if you want to utilize GPUs.")
    
    # Python version and executable
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"NumPy Version: {np.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    try:
        import abtem
        print(f"abTEM Version: {abtem.__version__}")
    except ImportError:
        print("Didn't find abTEM")
    try:
        import cupy
        print(f"Cupy Version: {cupy.__version__}")
    except ImportError:
        print("Didn't find Cupy")
    print(" ")
    
def time_sync():
    # PyTorch doesn't have a direct exposed API to check the selected default device 
    # so we'll be checking these .is_available() just to prevent error.
    # Luckily these checks won't really affect the performance.
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Check if MPS (Metal Performance Shaders) is available (macOS only)
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    # Measure the time
    t = perf_counter()
    return t

def normalize(img):
    return (img-np.min(img))/(np.max(img) - np.min(img))