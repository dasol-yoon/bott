#!/usr/bin/env python3
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
An implementation of the Thompson sampling acquisition function for function networks.
"""
from typing import Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import Model
from botorch.utils.gp_sampling import get_gp_samples
from torch import Tensor


class ThompsonSampling(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
    ) -> None:
        r"""The Thompson sampling acquisition function for function networks.

        Args:
            model: A fitted GP network model.
            objective: An objective for the problem.
            
        Returns:
            None.
        """
        super(AcquisitionFunction, self).__init__()
        self.model = model
        self.GP_samples = get_gp_samples(
                model=model, num_outputs=1, n_samples=1
            )

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the acquisition value for function network on the candidate set `X`.

        Args:
            X: input tensor of shape `batch_shape x q x d` to evaluate the function network at.
        
        Returns:
            A  `batch_shape`-dim tensor of acquisition values.
        """
        output = self.GP_samples.posterior(X).mean
        while output.ndim != 1:
            output = output.squeeze(-1)
        return output