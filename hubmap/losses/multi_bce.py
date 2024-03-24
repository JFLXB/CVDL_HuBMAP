from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiOutputBCELoss(nn.Module):
    def __init__(
        self,
        weights: Tuple[float, float, float] = [1.0, 1.0, 1.0],
        interpolation_strategy: str = "bilinear",
    ):
        super().__init__()
        self._weights = weights
        self._interpolation_strategy = interpolation_strategy

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input1, input2, final = input
        weight1, weight2, weight_final = self._weights

        # We need to interplote the target to the respective size of the input
        target1 = F.interpolate(
            target, size=input1.size()[2:], mode=self._interpolation_strategy
        )
        target2 = F.interpolate(
            target, size=input2.size()[2:], mode=self._interpolation_strategy
        )

        l1 = F.binary_cross_entropy(input1, target1)
        l2 = F.binary_cross_entropy(input2, target2)
        lf = F.binary_cross_entropy(final, target)

        return weight1 * l1 + weight2 * l2 + weight_final * lf