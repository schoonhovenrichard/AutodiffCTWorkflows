import torch
from autodiffCT.operator import DifferentiableOperator


class VarianceLossFn(DifferentiableOperator):
    implements_batching = False
    def __init__(self, device=None):
        super().__init__(device=device)

    def __call__(self, input):
        input = input.to(self.device)
        return -input.var()


class AcutanceLossFn(DifferentiableOperator):
    implements_batching = False

    def __init__(self, device=None):
        super().__init__(device=device)

    def __call__(self, input):
        input = input.to(self.device).squeeze()
        partial_derivatives = torch.gradient(input)
        grad_magnitude_squared = sum([x**2 for x in partial_derivatives])
        grad_l2_norm = torch.sqrt(grad_magnitude_squared.sum())
        return -grad_l2_norm / grad_l2_norm.nelement()
