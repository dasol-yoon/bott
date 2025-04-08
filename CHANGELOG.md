# Changelog

All notable changes to this project will be documented in this file.

## 2025.04.08
- Rename modules into `physics_models` and `problem`
- Add `run_bott.ipynb` notebook for temporarily development
- Add `CHANGELOG.md` to log the changes and archive todos/ideas
- Add `.gitignore` to allow keeping some large files untracked by Git
- Modify `__init_.py` by commenting out all import and add a `__version__` variable
- 

# Note
- In physics community we usually do f(x) = y, here y is our diffraciton patterns. Since we have a physics output and a model objective, it might be cleaner to reserve all the "y" for diffraction pattern, and use explictly "loss" or "objective" for the surrogate model output. "objective" might be better for optimization community.

# Package architecture as of 2025.04.08
- __init__y     : keeps the version
- io            : all the loading/saving related operations, expected to be very minimal
- metrics       : the loss and the preprocessing/reduction methods before calculating loss (tiling in our case)
- optimization  : contains wrapper function for the optimization task (run trial, get new sample)
- physics_models: variants of physical forward model like f(thickness, tilt) = PACBED
- problem       : core class that combines physics model, tiling, loss into a test function. problem.evaluate() = objective 
- utils         : everything that fall out of other modules
- visualization : everything about plotting 

The workflow would be calling the wrapper function in `optimization`, which calls the `problem` class. The `problem` class would have the needed components (physics model, tilting function, loss function) initiated and return the objective value so the wrapper function (`run_one_trial`) can complete 1 trial.

Hopefully we can make `optimization` and `problem` to be entirely task-agnostic, and let it focus only on x, y, and objective. So that they can work on something beyond PACBED calibration. `physics_models` currently considers only f(thickness, tilt) = PACBED, but we can freely add more physics models to include other target parameters or different type of forward process.

When we are about to add "gradient descent" or "classical BO" or other optimization strategy, they would probably get into `optimization` module. A cleaner solution is to make a `OptimizationTask` class, and implement the specific routine for strategies like "gradient descent", "classical BO", "composite BO".