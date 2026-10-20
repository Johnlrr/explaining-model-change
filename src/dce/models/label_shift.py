"""v1.10 T-A: post-hoc label-shift update (Saerens et al.): f_new = f_old with b added to the class-1 logit.

With m = f_y − f_{1−y} on logits (dce.models.cnn.margin), m_new = m_old + b for y = 1 and m_old − b for y = 0, at
every input. So h = m_new − m_old is constant, ∇m_new ≡ ∇m_old, and B3 is undefined (‖g‖ = 0). b > 0 lowers the
class-0 margin: type-N witnesses can exist only for y = 0 queries; b < 0 only for y = 1.
"""

from __future__ import annotations

import torch
from torch import nn


class LogitShift(nn.Module):
    """Wraps a frozen classifier and adds the constant b to the logit of class 1."""

    def __init__(self, base: nn.Module, b: float):
        super().__init__()
        self.base = base
        self.b = float(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        return logits + torch.tensor([0.0, self.b], dtype=logits.dtype, device=logits.device)


def lowered_class(b: float) -> int:
    """The class whose margin the shift lowers (type-N witnesses exist only for queries of this class)."""
    return 0 if b > 0 else 1
