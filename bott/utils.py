# utils are for helper functions that don't fit elsewhere

def normalize_arr(arr, norm_type='zero_to_one'):
    
    if norm_type == 'zero_to_one':
        norm_arr = (arr - arr.min()) / (arr.max() - arr.min())
    
    elif norm_type == 'standardize':
        norm_arr = (arr - arr.mean()) / arr.std()
    
    else:
        raise KeyError(f"norm_type {norm_type} not implemented yet, please use 'zero_to_one', or 'standardize'!")
    
    return norm_arr

def get_default_abTEM_params():
    params_abTEM = {
        
    }
    return params_abTEM