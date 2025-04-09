# Changelog

All notable changes to this project will be documented in this file.

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
- `reduction` needs more work to refine the reduction methods
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