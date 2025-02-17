# BOTT (Bayesian Optimized Tilt & Thickness)

Use Bayesian Optimization to get thickness and tilt of the given position averaged convergent beam electron diffraction (PACBED) image. 

This package uses abTEM to simulate PACBEDs, which are used as input images during the Bayesian Optimization.

1. Create a python environment for just this project.
2. GPU for faster PACBED simulations in abTEM: If you have an NVIDIA GPU, check the version of cudatoolkit and install the right version of pytorch accordingly.<br>
   ex) `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`
3. Install other packages in the following order. (Installation of pacakages with different versions/orders could mess up the compatibility. Known bugs are jupyterlab or matplotlib not working).<br>
    `conda install -c conda-forge jupyterlab botorch gpytorch matplotlib`<br>
    Install abtem-legacy(https://github.com/abTEM/abTEM-legacy) using git or from a zip file.<br>
    `conda install pandas`

Possible alternative to the steps 2-3: Install packages using `environments.txt`.
    
[This notebook (BO_1211_preloaded...)](https://github.coecis.cornell.edu/dy327/ttBO/blob/main/BO_1211_preloaded%2Btilt_integer-debugPlotting.ipynb) has both tilt and thickness implemented, but uses pre-simulated PACBEDs. It does not require the use of abTEM. Check [this Box folder](https://cornell.box.com/s/s36n24d59rjqw2ibejxn0ta40lx6jr02) for the dataset.<br>
[This notebook (BO_240118)](https://github.coecis.cornell.edu/dy327/ttBO/blob/main/BO_240118.ipynb) has thickness implemented for an experimental PACBED.
