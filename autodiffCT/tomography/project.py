from copy import deepcopy

import torch
from tomosipo.Operator import Operator as TomosipoProjector
from tomosipo.torch_support import OperatorFunction, to_autograd

from autodiffCT.operator import DifferentiableOperator
from autodiffCT.parameter import TorchParameter


class AutogradedTomosipoOperator(TomosipoProjector):
    def __init__(self, tomosipo_operator):
        self.tomosipo_operator = deepcopy(tomosipo_operator)

    @property
    def implements_batching(self):
        return True

    def __getattr__(self, attr_name):
        if 'tomosipo_operator' not in self.__dict__:
            raise AttributeError(f'tomosipo_operator attribute is not initialized.')
        return getattr(self.tomosipo_operator, attr_name)

    def __call__(self, x):
        return OperatorFunction.apply(x, self.tomosipo_operator)

    def transpose(self):
        return AutogradedTomosipoOperator(self.tomosipo_operator.T)


class TomosipoOperatorMixin(DifferentiableOperator):
    def __init__(self, projector, device=None, **kwargs):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        super().__init__(device=device, **kwargs)

        if isinstance(projector, AutogradedTomosipoOperator):
            self.projector = projector
        elif isinstance(projector, TomosipoProjector):
            self.projector = AutogradedTomosipoOperator(projector)
        else:
            raise ValueError('Projector expected to be a Tomosipo operator. '
                             f'Instead got {type(projector)}.')

        if self.projector.domain_shape[0] == 1:
            self.is_2d = True
        else:
            self.is_2d = False

    @property
    def output_shape(self, input_dims):
        if self.is_2d:
            return self.projector.range_shape[1], self.projector.range_shape[2]
        else:
            return tuple(self.projector.range_shape)

    @property
    def input_shape(self):
        if self.is_2d:
            return self.projector.domain_shape[1], self.projector.domain_shape[2]
        else:
            return tuple(self.projector.domain_shape)

    def _preprocess(self, data):
        data = data.to(self.device)

        # Tomosipo projector works on 3D data so 2D is passed as a volume with 1 slice.
        if self.implements_batching:
            if self.is_2d and data.ndim == 3:
                data = data.unsqueeze(dim=1)
        else:
            if self.is_2d and data.ndim == 2:
                data = data.unsqueeze(dim=0)
        return data

    def _postprocess(self, data):
        # Remove single slice added for 3D tomosipo projector.
        if self.implements_batching:
            if self.is_2d and data.ndim == 3:
                data = data.squeeze(dim=1)
        else:
            if self.is_2d and data.ndim == 2:
                data = data.squeeze(dim=0)
        return data


class ProjectOperator(TomosipoOperatorMixin, DifferentiableOperator):
    def __init__(self, projector, device=None):
        super().__init__(projector=projector, device=device)

    @property
    def implements_batching(self):
        return True

    def __call__(self, vol_data):
        vol_data = self._preprocess(vol_data)
        proj_data = self.projector(vol_data)
        proj_data = self._postprocess(proj_data)
        return proj_data


class BackprojectOperator(TomosipoOperatorMixin, DifferentiableOperator):
    def __init__(self, projector, device=None):
        super().__init__(projector=projector, device=device)

    @property
    def implements_batching(self):
        return True

    def __call__(self, proj_data):
        proj_data = self._preprocess(proj_data)
        vol_data = self.projector.T(proj_data)
        vol_data = self._postprocess(vol_data)
        return vol_data
