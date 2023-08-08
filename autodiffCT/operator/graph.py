import torch

from autodiffCT.operator import BaseOperator, DifferentiableOperator


class VoidOperator(BaseOperator):
    def __init__(self):
        super().__init__(device='cpu')

    @property
    def implements_batching(self):
        return True

    def __call__(self, data=None):
        return data

    @property
    def output_shape(self, input_dims):
        return (float('inf'),)

    @property
    def input_shape(self):
        return (float('inf'),)


class MatchScalingOperator(BaseOperator):
    def __init__(self):
        super().__init__(device='cpu')

    @property
    def implements_batching(self):
        return False

    def __call__(self, data):
        fixed_tensor = data[0,...]
        tensor_to_scale = data[1,...]
        if tensor_to_scale.std() != 0:
            tensor_to_scale = (tensor_to_scale / tensor_to_scale.std())
        tensor_to_scale = tensor_to_scale - tensor_to_scale.mean()
        tensor_to_scale = tensor_to_scale * fixed_tensor.std()
        tensor_to_scale = tensor_to_scale + fixed_tensor.mean()
        return tensor_to_scale

    @property
    def output_shape(self, input_dims):
        return (float('inf'),)

    @property
    def input_shape(self):
        return (float('inf'),)
