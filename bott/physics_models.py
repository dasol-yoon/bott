# Put the forward methods (like f(tilt, thickness) = PACBED)
# We may turn this into a class if we have other physics models
from bott.utils import get_default_abTEM_params

def simulate_potential(thickness, params_abTEM=None):
    '''
    This one is for future gradient descent purpose
    The potential will be returned, and we'll generate the measurement based on some in-house multislice simulation code with tilted propagator
    
    Note that the underlying potential array for this GD approach is fixed, and we're using GD to optimizer the tilted propafator parameterized by slice thickness.
    Therefore, the computation complexity is fixed for the GD approach and our achievable thickness range could be a bit limited.
    while the `simulate_cbed` would have the actual potential array with the right thickness and number of slices.
    For fair comparison, we might want to fix a thickness and just do tilt optimization.
    '''
    
    if params_abTEM is None:
        params_abTEM = get_default_abTEM_params()
     
    # Unpack params_abTEM explictly
    device_abtem = params_abTEM["device_abtem"]
    path_crystal = params_abTEM["path_crystal"]

    potential_extent_x = params_abTEM["potential_extent_x"]
    potential_extent_y = params_abTEM["potential_extent_y"]
    lateral_sampling = params_abTEM["lateral_sampling"]
    vertical_sampling = params_abTEM["vertical_sampling"]
    potential_parametrization = params_abTEM["potential_parametrization"]
    potential_projection = params_abTEM["potential_projection"]
    
    random_seed = params_abTEM["random_seed"]
    use_frozen_phonon = params_abTEM["use_frozen_phonon"]
    num_phonon_configs = params_abTEM["num_phonon_configs"]
    phonon_sigma = params_abTEM["phonon_sigma"]
    
    # Note that the printing are commented out
    
    # Setup imports
    import ase
    import abtem
    import numpy as np
    import dask

    if device_abtem == 'gpu':
        import cupy as xp
        abtem.config.set({"dask.chunk-size-gpu" : "2048 MB"})
        dask.config.set({"num_workers": 1})
    elif device_abtem == 'cpu':
        import numpy as xp
    else:
        raise ValueError(f"device_abtem '{device_abtem}' not implemented yet, please use 'cpu', or 'gpu'!")

    # abtem configure
    abtem.config.set({"local_diagnostics.progress_bar": False})
    abtem.config.set({"device": device_abtem})
    
    # Setup cell
    unit_cell = ase.io.read(path_crystal)
    target_object_extent = np.array((potential_extent_x, potential_extent_y, thickness)) # Specimen range in Ang (x,y,z)
    cell_constants = np.diag(unit_cell.cell)
    super_cell_reps = np.ceil(target_object_extent / cell_constants).astype('int')
    super_cell = unit_cell * super_cell_reps
    # print(f"super_cell = {super_cell}")
    
    # Calculate the potential
    if use_frozen_phonon:
        # print(f"Using FrozenPhonons potential with {num_phonon_configs} configs")
        atoms = abtem.FrozenPhonons(atoms=super_cell, num_configs=num_phonon_configs, sigmas=phonon_sigma, seed=random_seed)
        potential = abtem.Potential(atoms=atoms, sampling=lateral_sampling, parametrization=potential_parametrization,
            slice_thickness=vertical_sampling, projection=potential_projection)
        potential_arr = xp.mean(potential.build().compute(progress_bar=False).array, axis=0)
    else:
        # print("Using Static potential")
        potential = abtem.Potential(atoms=super_cell, sampling=lateral_sampling, parametrization=potential_parametrization,
            slice_thickness=vertical_sampling, projection=potential_projection)
        potential_arr = potential.build().compute(progress_bar=False).array
    # print(f"potential.shape = {potential.shape}")
    
    # Cast potential_arr into numpy array
    if device_abtem == 'gpu':
        potential_arr = potential_arr.get()
    
    return potential_arr
    
def simulate_cbed(thickness, tilt_x, tilt_y, params_abTEM=None):
    
    if params_abTEM is None:
        params_abTEM = get_default_abTEM_params()
     
    # Unpack params_abTEM explictly
    device_abtem = params_abTEM["device_abtem"]
    path_crystal = params_abTEM["path_crystal"]

    potential_extent_x = params_abTEM["potential_extent_x"]
    potential_extent_y = params_abTEM["potential_extent_y"]
    lateral_sampling = params_abTEM["lateral_sampling"]
    vertical_sampling = params_abTEM["vertical_sampling"]
    potential_parametrization = params_abTEM["potential_parametrization"]
    potential_projection = params_abTEM["potential_projection"]

    random_seed = params_abTEM["random_seed"]
    use_frozen_phonon = params_abTEM["use_frozen_phonon"]
    num_phonon_configs = params_abTEM["num_phonon_configs"]
    phonon_sigma = params_abTEM["phonon_sigma"]

    energy = params_abTEM["energy"]
    convergence_angle = params_abTEM["convergence_angle"]
    df = params_abTEM["df"]
    aberrations = params_abTEM["aberrations"]

    scan_step_size = params_abTEM["scan_step_size"]
    return_pacbed = params_abTEM["return_pacbed"]
    
    
    ########################################################################################################################
    
    # Note that the printing are commented out
    
    # Setup imports
    import ase
    import abtem
    import numpy as np
    import dask

    if device_abtem == 'gpu':
        import cupy as xp
        abtem.config.set({"dask.chunk-size-gpu" : "2048 MB"})
        dask.config.set({"num_workers": 1})
    elif device_abtem == 'cpu':
        import numpy as xp
    else:
        raise ValueError(f"device_abtem '{device_abtem}' not implemented yet, please use 'cpu', or 'gpu'!")

    # abtem configure
    abtem.config.set({"local_diagnostics.progress_bar": False})
    abtem.config.set({"device": device_abtem})
    
    # Setup cell
    unit_cell = ase.io.read(path_crystal)
    target_object_extent = np.array((potential_extent_x, potential_extent_y, thickness)) # Specimen range in Ang (x,y,z)
    cell_constants = np.diag(unit_cell.cell)
    super_cell_reps = np.ceil(target_object_extent / cell_constants).astype('int')
    super_cell = unit_cell * super_cell_reps
    # print(f"super_cell = {super_cell}")
    
    # Calculate the potential
    if use_frozen_phonon:
        # print(f"Using FrozenPhonons potential with {num_phonon_configs} configs")
        atoms = abtem.FrozenPhonons(atoms=super_cell, num_configs=num_phonon_configs, sigmas=phonon_sigma, seed=random_seed)
        potential = abtem.Potential(atoms=atoms, sampling=lateral_sampling, parametrization=potential_parametrization,
            slice_thickness=vertical_sampling, projection=potential_projection)
    else:
        # print("Using Static potential")
        potential = abtem.Potential(atoms=super_cell, sampling=lateral_sampling, parametrization=potential_parametrization,
            slice_thickness=vertical_sampling, projection=potential_projection)
    # print(f"potential.shape = {potential.shape}")
    
    # Calculate the probe
    probe = abtem.Probe(energy=energy, semiangle_cutoff=convergence_angle, defocus=df, tilt=(tilt_x, tilt_y), **aberrations)
    probe.grid.match(potential)
    # # Useful information
    # from bott.utils import get_EM_constants
    # wavelength = get_EM_constants(energy/1e3, 'wavelength')
    # kmax_antialias = 1/lateral_sampling/3 # 1/Ang #The kmax_antialiasing = 2.675 Ang-1 
    # alpha_max_antialias = wavelength * kmax_antialias # rad
    # print(f"Energy = {energy/1e3} kV, rel. wavelength = {wavelength:.4f} Ang")
    # print(f"CBED collection kmax = {kmax_antialias:.4f} 1/Ang, collection alpha_max = {alpha_max_antialias*1000:.4f} mrad")
    # print(f"probe.shape = {probe.shape}")
    # print(f"probe.axes_metadata = {probe.axes_metadata}")
    
    
    # Make scan for 1 unit cell along x, y
    potential_extent = np.array(potential.extent)
    scan_start = np.array(potential_extent)/2
    scan_end = scan_start + cell_constants[:2]
    grid_scan = abtem.scan.GridScan(start=scan_start, end=scan_end,sampling=scan_step_size)
    # print(f"grid_scan.axes_metadata = {grid_scan.axes_metadata}")
    
    # Get cbeds
    cbeds = probe.multislice(scan = grid_scan, potential = potential).diffraction_patterns(max_angle='cutoff').reduce_ensemble().compute(progress_bar=False)
    # print(f"cbeds.axes_metadata = {cbeds.axes_metadata}")

    if return_pacbed:
        measurement = xp.mean(cbeds.array, axis=(0,1))
    else:
        measurement = cbeds.array

    # Cast measurement into numpy array
    if device_abtem == 'gpu':
        measurement = measurement.get()
    
    return measurement