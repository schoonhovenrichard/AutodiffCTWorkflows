import torch
import torch.nn as nn

from autodiffCT.operator import BaseOperator, DifferentiableOperator
from autodiffCT.parameter import TorchParameter


class ClearPyTorchGpuCacheOperator(BaseOperator):
    def __init__(self):
        super().__init__()
        self.parameters = {}

    @property
    def implements_batching(self):
        return True

    def __call__(self, data):
        torch.cuda.empty_cache()
        return data

    @property
    def output_shape(self, input_dims):
        return input_dims

    @property
    def input_shape(self):
        return (float('inf'),)


class Scaling3dOperator(DifferentiableOperator):
    def __init__(self, c_in, c_out, init_data=None, learn=True, device='cpu'):
        super().__init__(device=device)

        self.module = nn.Conv3d(c_in, c_out, 1)
        with torch.no_grad():
            self.module.weight = nn.Parameter(torch.ones_like(self.module.weight))
            self.module.bias = nn.Parameter(torch.zeros_like(self.module.bias))

        if init_data is not None:
            with torch.no_grad():
                mean_in = init_data.mean()
                square_in = init_data.pow(2).mean()
                std_in = torch.sqrt(square_in - mean_in ** 2)

                self.module.weight = nn.Parameter(torch.zeros_like(self.module.weight))
                for i in range(self.module.weight.shape[0]):
                    self.module.weight[i, i] = 1/std_in
                self.module.bias.data.fill_(-mean_in / std_in)

        if not learn:
            self.module.bias.requires_grad = False
            self.module.weight.requires_grad = False

        self.module = self.module.to(device)
        self.parameters = {}
        for i, param in enumerate(self.module.parameters()):
            self.parameters[f'param{i}'] = TorchParameter(param,
                                                          device=param.device,
                                                          copy=False)
        self.weights = []
        self.biases = []

    @property
    def implements_batching(self):
        return True

    def __call__(self, data):
        self.weights.append(self.module.weight.data[0,0,0,0].item())
        self.biases.append(self.module.bias.data[0].item())

        if data.ndim == 3:
            return self.module(data[None,None,...]).squeeze(1)
        elif data.ndim == 4:
            return self.module(data[:,None,...]).squeeze(1)
            ret_tens = self.module(data[:,None,...])
            ret_tens = ret_tens.squeeze(1)
            return ret_tens
        elif data.ndim == 6:
            if data.shape[1] != 1:
                raise Exception("Unable to process 4 dimensional "
                                "reconstruction data with no singleton second "
                                "dimension")
            return self.module(data[:,0,...]).squeeze(1)
        elif data.ndim == 5:
            return self.module(data).squeeze(1)

    def get_output_dimensions(self, input_dims):
        arr = torch.zeros(input_dims,
                          device=next(self.module.parameters()).device)
        returned_arr = self.__call__(arr)
        return returned_arr.size()

    def required_input_dimensions(self):
        return (float('inf'),)
