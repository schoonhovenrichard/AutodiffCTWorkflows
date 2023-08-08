import copy
import warnings

import torch


class BaseParameter:
    def __new__(cls, value):
        if isinstance(value, BaseParameter):
            parameter = value
        else:
            parameter = super().__new__(cls)
            parameter.value = copy.deepcopy(value)
        return parameter

    def __getnewargs__(self):
        # This is necessary to allow pickle to correctly load parameters.
        return (self.value,)

    def __repr__(self):
        return 'Parameter(%s)' % self.value


class NumericalParameter(BaseParameter):
    pass


class IntegerParameter(NumericalParameter):
    pass


class OrdinalParameter(BaseParameter):
    pass


class CategoricalParameter(BaseParameter):
    pass


class TorchParameter(NumericalParameter):
    def __new__(cls, value, copy_if_torch_tensor=True, **kwargs):
        if isinstance(value, cls):
            # Store by reference if an already existing parameter is passed
            if kwargs:
                warnings.warn('Attempting to copy parameter by reference '
                              '(existing parameter passed as input) but '
                              'additional keyword arguments to `torch.Tensor` '
                              'constructor were also specified. These will be '
                              'ignored.')
            parameter = value
        elif not copy_if_torch_tensor:
            parameter = object.__new__(cls)
            parameter.value = value
        else:
            parameter = super().__new__(cls, value)
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
            device = kwargs.pop('device', device)
            dtype = kwargs.pop('dtype', None)
            requires_grad = kwargs.pop('requires_grad', True)
            if isinstance(parameter.value, torch.Tensor):
                parameter.value = parameter.value.to(device, dtype) \
                                                 .requires_grad_(requires_grad)
            else:
                parameter.value = torch.tensor(parameter.value, device=device,
                                               dtype=dtype, requires_grad=requires_grad)

        return parameter

    def to_device(self, device):
        if isinstance(self.value, torch.nn.Module):
            self.value.to(device)
        elif isinstance(self.value, torch.Tensor):
            self.value = self.value.to(device)
