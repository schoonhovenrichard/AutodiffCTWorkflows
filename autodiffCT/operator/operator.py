import copy
from abc import ABC, abstractmethod

import dill as pickle
import torch

from autodiffCT.parameter import TorchParameter


class BaseOperator(ABC):
    def __init__(self, device='cpu'):
        self.parameters = {}
        self.device = device

    @abstractmethod
    def __call__(self):
        pass

    @property
    @abstractmethod
    def implements_batching(self):
        pass

    @property
    def output_shape(self, input_dims):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    def save(self, path):
        to_save = copy.deepcopy(self)
        to_save.to_device('cpu')
        state = to_save.get_state()
        state['device'] = self.device
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.set_state(state)
        if state['device'] != 'cpu':
            self.to_device(state['device'])

    def to_device(self, device):
        self.device = device
        for attribute in self.__dict__.values():
            if isinstance(attribute, TorchParameter):
                attribute.to_device(device)
            elif isinstance(attribute, torch.nn.Module):
                attribute.to(device)
            elif isinstance(attribute, torch.Tensor):
                attribute = attribute.to(device)

    def set_state(self, state):
        self.__dict__.update(state)

    def get_state(self):
        return self.__dict__


class DifferentiableOperator(BaseOperator):
    pass


class ConvexOperator(DifferentiableOperator):
    pass
