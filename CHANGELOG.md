# Changelog

All notable changes to this project will be documented in this file.

## 2025.04.13
- Fix the "EICF" algo by allowing objective lamda to take `X=None` becuase it's strictly needed for botorch 0.13.0
- Worked through the `device` for major components but seems like BO runs slower on GPU with botorch (see `runner.ipynb`)
- Add `environment.yml`; Update `README.md` for installation instruction.

## 2025.04.12
- @PB hook up the `optimization.py` and `runner.ipynb`
- Pull `abtem_device` out of `params_abtem` dict so we can specify it outside, default currently set as "cpu". Also rename the `abTEM` into `abtem` for simplicity.
- Add `reduce` option to loss functions so we can optionally keep the first (batch) dimension
- Specify `dtype` and `device` for `problem.py` so we can easily switch between CPU/GPU and data type.
- Modify `reduction.py` so all tiling functions are supporting (B,H,W) 3D input tensors. (B,H,W) -> (B, mxn, H/m, W/n)
- Polish `optimization.py` to make consistent tensor construction with device/dtype, use batch processing, improve comments

## 2025.04.10
- Add `figure_full_abTEM_simulation_timing` and `figure_potential_generation_timing` notebooks
- Initiate the notebook for GD and AD-based multislice development

## 2025.04.09
- Refine `reduction` module. Generalize the vert/hori tiling into `get_long_tiles`. Rename `domainKnowledgeTile` into `circular_tiles` for clarity. Enable overlapping tiles by specifying "tile_width" and allowing padding the measurement. Enable different reduce methods including "mean", "sum", and "False/None" to return the original tiles for debugging purpose.
- Modify `OptimizationProblem` so it can correctly initialized with the `super().__init__()`.
- Change the `ground_truth_path` to `ground_truth` for more flexibility
- Implictly assume input `X` is a size 1 tensor of (3,), but we can decide whether we want it that way or not, just needs to be consistent.
- Shuffle the order of fields for output file name so it's "Thickness_0200_TiltX_15_TiltY_16.tif" just like the order for `simulate_cbed`

## 2025.04.08
- Rename modules into `physics_models` and `problem`
- Add `run_bott.ipynb` notebook for temporarily development
- Add `CHANGELOG.md` to log the changes and archive todos/ideas
- Add `.gitignore` to allow keeping some large files untracked by Git
- Modify `__init_.py` by commenting out all import and add a `__version__` variable
- Refactor the `simulate_cbed` function and make it cpu/gpu compatible
- Split `metrics` module into `loss` and `reduction` for clarity
- Add `simulate_potential` in case we're doing GD in the future
- Fill in `loss` and `reduction` modules

# Note
- I ended up choosing to use "measurement" for diffraction patterns, "y" for the reduced diffraction (so it's cleaner in the `loss`), and "loss" for the loss output. From the surrogate model end we will just call it "objective". Like surrogate_model.get_objective(), and def get_objective() return loss.

# TODO
- `loss` might need some discussion of whether we need to return a list or not
- Need to make sure we have the correct normalization and resampling for both the measurement_true and measurement_simu

# Package architecture as of 2025.04.08
- __ init __  : keeps the version
- io            : all the loading/saving related operations, expected to be very minimal
- loss          : the LossFunction class
- optimization  : contains wrapper function for the optimization task (run trial, get new sample)
- physics_models: variants of physical forward model like f(thickness, tilt) = PACBED. Could change into a class if we have more physics model in the future
- problem       : core class that combines physics model, tiling, loss into a test function. problem.evaluate() = objective
- reduction     : the ReductionFunction class for preprocessing needed before calculating loss (tiling in our case)
- utils         : everything that fall out of other modules
- visualization : everything about plotting 

The workflow would be calling the wrapper function in `optimization`, which calls the `problem` class. The `problem` class would have the needed components (physics model, reduction function, loss function) initiated and return the objective value so the wrapper function (`run_one_trial`) can complete 1 trial.

Hopefully we can make `optimization` and `problem` to be entirely task-agnostic, and let it focus only on x, y, and objective. So that they can work on something beyond PACBED calibration. `physics_models` currently considers only f(thickness, tilt) = PACBED, but we can freely add more physics models to include other target parameters or different type of forward process.

When we are about to add "gradient descent" or "classical BO" or other optimization strategy, they would probably get into `optimization` module. A cleaner solution is to make a `OptimizationTask` class, and implement the specific routine for strategies like "gradient descent", "classical BO", "composite BO".