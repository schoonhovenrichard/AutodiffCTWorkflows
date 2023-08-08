from autodiffCT.operator import BaseOperator#, DifferentiableOperator


class ApplyRoiOperator(BaseOperator):
    implements_batching = False

    def __init__(self, roi, device=None):
        super().__init__(device)
        self.roi = roi

    def __call__(self, input):
        input = input.to(self.device)
        return input[self.roi]
