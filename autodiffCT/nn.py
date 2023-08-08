import torch

from autodiffCT.operator import DifferentiableOperator
from autodiffCT.parameter import TorchParameter


class PyTorchNn2dOperator(DifferentiableOperator):
    def __init__(self, model, device=None):
        """Wrapper around nn.Sequential object of PyTorch."""
        if device is None:
            device = next(model.parameters()).device
            self.model = model
        else:
            self.model = model.to(device)
        super().__init__(device=device)
        self.parameters = {}
        for i, param in enumerate(self.model.parameters()):
            self.parameters[f'weights{i}'] = TorchParameter(param, copy_if_torch_tensor=False)

    @property
    def implements_batching(self):
        return True

    def __call__(self, data):
        data = self._preprocess(data)
        return self._postprocess(self.model(data))

    @property
    def output_shape(self, input_dims):
        arr = torch.zeros(input_dims,
                          device=next(self.model.parameters()).device)
        returned_arr = self.__call__(arr)
        return returned_arr.size()

    @property
    def input_shape(self):
        return (None, self.parameters[0].weight.size[1], None)

    def _preprocess(self, data):
        data = data.to(self.device)
        if data.ndim == 3:
            data = data.unsqueeze(dim=1) # Add channel dimension
        return data

    def _postprocess(self, data):
        data = data.squeeze(dim=1)
        return data
