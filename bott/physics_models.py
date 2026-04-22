# Put the forward methods (like f(tilt, thickness) = PACBED)
# We may turn this into a class if we have other physics models
from bott.utils import get_default_abtem_params, normalize


def simulate_potential(thickness, params_abtem=None, device_simu='cpu'):
    '''
    This one is for future gradient descent purpose
    The potential will be returned, and we'll generate the measurement based on some in-house multislice simulation code with tilted propagator
    
    Note that the underlying potential array for this GD approach is fixed, and we're using GD to optimizer the tilted propafator parameterized by slice thickness.
    Therefore, the computation complexity is fixed for the GD approach and our achievable thickness range could be a bit limited.
    while the `simulate_cbed` would have the actual potential array with the right thickness and number of slices.
    For fair comparison, we might want to fix a thickness and just do tilt optimization.
    '''
    
    if params_abtem is None:
        params_abtem = get_default_abtem_params()
     
    # Unpack params_abtem explictly
    path_crystal = params_abtem["path_crystal"]

    potential_extent_x = params_abtem["potential_extent_x"]
    potential_extent_y = params_abtem["potential_extent_y"]
    lateral_sampling = params_abtem["lateral_sampling"]
    vertical_sampling = params_abtem["vertical_sampling"]
    potential_parametrization = params_abtem["potential_parametrization"]
    potential_projection = params_abtem["potential_projection"]
    
    random_seed = params_abtem["random_seed"]
    use_frozen_phonon = params_abtem["use_frozen_phonon"]
    num_phonon_configs = params_abtem["num_phonon_configs"]
    phonon_sigma = params_abtem["phonon_sigma"]
    
    # Note that the printing are commented out
    
    # Setup imports
    import abtem
    import ase
    import dask
    import numpy as np

    if device_simu == 'gpu':
        import cupy as xp
        abtem.config.set({"dask.chunk-size-gpu" : "2048 MB"})
        dask.config.set({"num_workers": 1})
    elif device_simu == 'cpu':
        import numpy as xp
    else:
        raise ValueError(f"device_simu '{device_simu}' not implemented yet, please use 'cpu', or 'gpu'!")

    # abtem configure
    abtem.config.set({"local_diagnostics.progress_bar": False})
    abtem.config.set({"device": device_simu})
    
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
    if device_simu == 'gpu':
        potential_arr = potential_arr.get()
    
    return potential_arr
    
def simulate_cbed(thickness, tilt_x, tilt_y, params_abtem=None, 
                  device_simu='cpu', pbar=False, scan_coords=None): 
    #scan coords:0207temp
    
    if params_abtem is None:
        params_abtem = get_default_abtem_params()
     
    # Unpack params_abtem explictly
    path_crystal = params_abtem["path_crystal"]

    potential_extent_x = params_abtem["potential_extent_x"]
    potential_extent_y = params_abtem["potential_extent_y"]
    lateral_sampling = params_abtem["lateral_sampling"]
    vertical_sampling = params_abtem["vertical_sampling"]
    potential_parametrization = params_abtem["potential_parametrization"]
    potential_projection = params_abtem["potential_projection"]

    random_seed = params_abtem["random_seed"]
    use_frozen_phonon = params_abtem["use_frozen_phonon"]
    num_phonon_configs = params_abtem["num_phonon_configs"]
    phonon_sigma = params_abtem["phonon_sigma"]

    energy = params_abtem["energy"]
    convergence_angle = params_abtem["convergence_angle"]
    df = params_abtem["df"]
    aberrations = params_abtem["aberrations"]

    scan_step_size = params_abtem["scan_step_size"]
    return_pacbed = params_abtem["return_pacbed"]

    try: #may adjust this in the future
        if params_abtem['detector_angle']:
            detector_angle = params_abtem["detector_angle"]
    except KeyError:
        detector_angle = 'cutoff'
    
    ########################################################################################################################
    
    # Note that the printing are commented out
    
    # Setup imports
    import abtem
    import ase
    import dask
    import numpy as np

    if device_simu == 'gpu':
        import cupy as xp
        abtem.config.set({"dask.chunk-size-gpu" : "2048 MB"})
        dask.config.set({"num_workers": 1})
    elif device_simu == 'cpu' or device_simu == 'mps':
        import numpy as xp
    else:
        raise ValueError(f"device_simu '{device_simu}' not implemented yet, please use 'cpu', or 'gpu'!")

    # abtem configure
    abtem.config.set({"local_diagnostics.progress_bar": False})
    abtem.config.set({"device": device_simu})

    # Setup cell
    unit_cell = ase.io.read(path_crystal)
    target_object_extent = np.array((potential_extent_x, potential_extent_y, thickness)) # Specimen range in Ang (x,y,z)
    cell_constants = np.diag(unit_cell.cell)
    super_cell_reps = np.ceil(target_object_extent / cell_constants).astype('int')
    super_cell = unit_cell * super_cell_reps
    print(f"super_cell = {super_cell}")
    if params_abtem['vac_x']: #20250717 temp
        vac_x = params_abtem['vac_x']
        super_cell.center(axis=(0),vacuum=vac_x)
    
    # Calculate the potential
    if use_frozen_phonon:
        print(f"Using FrozenPhonons potential with {num_phonon_configs} configs")
        atoms = abtem.FrozenPhonons(atoms=super_cell, num_configs=num_phonon_configs, sigmas=phonon_sigma, seed=random_seed)
        potential = abtem.Potential(atoms=atoms, sampling=lateral_sampling, parametrization=potential_parametrization,
            slice_thickness=vertical_sampling, projection=potential_projection)
    else:
        print("Using Static potential")
        potential = abtem.Potential(atoms=super_cell, sampling=lateral_sampling, parametrization=potential_parametrization,
            slice_thickness=vertical_sampling, projection=potential_projection)
    # print(f"potential.shape = {potential.shape}")
    
    # Calculate the probe
    #20250604 temp: int added
    probe = abtem.Probe(energy=energy, semiangle_cutoff=convergence_angle, defocus=df, tilt=(int(tilt_x), int(tilt_y)), **aberrations)
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
    if scan_coords:
        scan_start = scan_coords[0]
        scan_end = scan_coords[1]
        print(scan_start)
        print(scan_end)
    else:
        potential_extent = np.array(potential.extent)
        scan_start = np.array(potential_extent)/2
        scan_end = scan_start + cell_constants[:2]
    grid_scan = abtem.scan.GridScan(start=scan_start, end=scan_end,sampling=scan_step_size)
    # print(f"grid_scan.axes_metadata = {grid_scan.axes_metadata}")
    
    # Get cbeds
    cbeds = probe.multislice(scan = grid_scan, potential = potential).diffraction_patterns(max_angle=detector_angle).reduce_ensemble().compute(progress_bar=pbar)
    # print(f"cbeds.axes_metadata = {cbeds.axes_metadata}")

    if return_pacbed:
        measurement = xp.mean(cbeds.array, axis=(0,1))
    else:
        measurement = cbeds.array

    # Cast measurement into numpy array
    if device_simu == 'gpu':
        measurement = measurement.get()
    
    return measurement