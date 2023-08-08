from abc import ABC, abstractmethod
from contextlib import nullcontext

import dill as pickle
import torch
from tqdm import trange


class BaseOptimizer(ABC):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @abstractmethod
    def optimize(self, data_loader, loss_func, target_loader=None):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    def save(self, path):
        state = self.get_state()
        with open(path, 'wb') as f:
            pickle.dump(state,f)

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.set_state(state)


class PyTorchOptimizer(BaseOptimizer):
    def __init__(self, pipeline, optimizer_class, target_parameters=None,
                 *args, **kwargs):
        super().__init__(pipeline)
        # target_parameters is either None (default for all available parameters),
        # an iterable of parameters, or # a dictionary of (operator, learning_rate).
        if target_parameters is None:
            target_parameters = pipeline.get_parameters()

        if isinstance(target_parameters, dict):
            # A separate learning rate needs to be set for operators in the
            # pipeline. The dictionary consists of keys (operators) and
            # values (learning rates).

            separate_lr_params = []
            for op, lr in target_parameters.items():
                if isinstance(lr, dict):
                    for param_key, separate_lr in lr.items():
                        target_param_dict = {}
                        param = op.parameters[param_key]
                        target_param_dict['params'] = param.value
                        target_param_dict['lr'] = separate_lr
                        separate_lr_params.append(target_param_dict)
                else:
                    for param in op.parameters.values():
                        target_param_dict = {}
                        target_param_dict['params'] = param.value
                        target_param_dict['lr'] = lr
                        separate_lr_params.append(target_param_dict)
            self.target_parameters = separate_lr_params
        else:
            self.target_parameters = [par.value for par in target_parameters]

        self.optimizer = optimizer_class(self.target_parameters, *args, **kwargs)
        self.losses = []

    def step(self, callback=None, constraints=None):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.losses.append(self.loss.detach().item())
        if callback is not None:
            callback(self)

        if constraints is not None:
            with torch.no_grad():
                for op, constr_params in constraints.items():
                    for param_key, limits in constr_params.items():
                        param = op.parameters[param_key]
                        req_grad = param.value.requires_grad
                        param.value.data = param.value.data.clamp(limits[0], limits[1])
                        param.value.requires_grad = req_grad

    def optimize(self, loss_func, inputs=None, targets=None, data_loader=None,
                 n_iterations=100, callback=None, save_gpu_memory=False, constraints=None):
        """Optimize pipeline using one of the algorithms from PyTorch."""
        with torch.autograd.graph.save_on_cpu(pin_memory=True) if save_gpu_memory else nullcontext():
            for self.iter_n in trange(n_iterations):
                if inputs is not None and targets is not None:
                    data_loader = zip(inputs, targets, strict=True)  # Zip-based iterators have
                    # to be re-created after they exhausted (i.e. at each iteration)
                if data_loader is not None:
                    for input, target in data_loader:
                        self.output = self.pipeline(input)
                        self.loss = loss_func(self.output, target)
                        self.step(callback=callback, constraints=constraints)
                elif inputs is not None:
                    for input in inputs:
                        self.output = self.pipeline(input)
                        self.loss = loss_func(self.output)
                        self.step(callback=callback, constraints=constraints)
                elif targets is not None:
                    self.output = self.pipeline()
                    for target in targets:
                        self.loss = loss_func(self.output, target)
                        self.step(callback=callback, constraints=constraints)
        return self.target_parameters

    def get_state(self):
        state = {
            "target_parameters": self.target_parameters,
            "losses": self.losses,
            "optimizer": self.optimizer
        }
        return state

    def set_state(self, state):
        self.target_parameters = state["target_parameters"]
        self.losses = state["losses"]
        self.optimizer = state["optimizer"]


class PyTorchSGD(PyTorchOptimizer):
    def __init__(self, pipeline, learning_rate, *args, **kwargs):
        super().__init__(pipeline, torch.optim.SGD, lr=learning_rate, *args, **kwargs)


class PyTorchAdam(PyTorchOptimizer):
    def __init__(self, pipeline, learning_rate, *args, **kwargs):
        super().__init__(pipeline, torch.optim.Adam, lr=learning_rate, *args, **kwargs)
