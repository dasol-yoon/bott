# Put the data loading and saving functions here
# Exp PACBED, cif, params files

import os
import numpy as np
from PIL import Image

def load_tif(file_path, verbose=False):
    from tifffile import imread

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    data = imread(file_path)
    if verbose:
        print("Success! Loaded .tif file path =", file_path)
        print("Imported .tif data shape =", data.shape)
    return data

def load_img(file_path, verbose=False):
    #load rgb images like png

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    with Image.open(file_path) as im:
        data = np.array(im.convert('L')) # convert to grayscale
    if verbose:
        print("Success! Loaded .img file path =", file_path)
        print("Imported .img data shape =", data.size)
    return data

def load_npy(file_path, verbose=False):

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    data = np.load(file_path)
    if verbose:
        print("Success! Loaded .npy file path =", file_path)
        print("Imported .npy data shape =", data.shape)
    return data

def load_pt(file_path, verbose=False):
    import torch

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

    data = torch.load(file_path)
    if verbose:
        print("Success! Loaded .pt file path =", file_path)
    return data

def load_yml(file_path, verbose=False):
    import yaml

    with open(file_path, "r") as file:
        params_dict = yaml.safe_load(file)
    params_dict['params_path'] = file_path

    if verbose:
        print("Success! Loaded .yml file path =", file_path)
    return params_dict