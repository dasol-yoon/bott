# BOTT (Bayesian Optimized Tilt & Thickness)

Use Bayesian Optimization to get thickness and tilt of the given position averaged convergent beam electron diffraction (PACBED) image. 

This package uses abTEM to simulate PACBEDs, which are used as input images during the Bayesian Optimization.

## Getting started
1. Download this GitHub repo to your local machine and unzip it.
2. Download and install a Python environment management software. For example, [miniforge](https://github.com/conda-forge/miniforge) is an open source, light weight alternative to [Anaconda](https://www.anaconda.com/download), which solves the environment much faster while only taking 1/10 of the disk space.
3. Run the following line to create an individual Python environment with all necessary dependencies assuming you have CUDA-supported NVIDIA GPU.

```bash
conda env create -f environment_bott_general.yml
```

Alternatively, you can also manually specify the packages for more flexibility (If your system does not have an nvidia GPU, remove these keywords(`cupy`, `pytorch-cuda=12.1`) from the following):
```bash
conda create -n bott python=3.11 pytorch=2.1.0 pytorch-cuda=12.1 botorch=0.13 cupy abtem gpytorch matplotlib tifffile ipykernel -c nvidia -c pytorch -c conda-forge
```

Note that your local CUDA version must be equal or higher than the specified `pytorch-cuda` version. You can check your local version by running
```bash
nvidia-smi
```

GPU can significantly accelerate the simulation speed of abTEM ofr PACBEDs.

PyTorch also supports Apple Silicon (MPS) on MacOS, but if you don't have GPU at all, you can still install the package with:
```bash
conda create -n bott python=3.11 pytorch=2.1.0 botorch=0.13 abtem gpytorch matplotlib tifffile ipykernel -c pytorch -c conda-forge
```
Note that the diffraction pattern simulation would be significantly slower on CPU.

4. Run the demo notebook in `scripts/run_bott.ipynb`

## Support
If you run into problems, have questions or suggested features / modifications, please create an issue [here](https://github.com/dasol-yoon/bott/issues/new/choose).

## Authors
- Dasol Yoon (dy327@cornell.edu)
- Poompol Buathong (pb482@cornell.edu)
- Chia-Hao Lee (cl2696@cornell.edu)

## License

## Citation
