import math

import numpy as np
import tomosipo
import torch
import torch.nn.functional as F
from torch.fft import irfft, rfft
from tqdm import trange
from ts_algorithms.fdk import fdk_weigh_projections

from autodiffCT.operator import DifferentiableOperator
from autodiffCT.parameter import TorchParameter
from autodiffCT.tomography.project import (AutogradedTomosipoOperator,
                                                TomosipoOperatorMixin)


class PyTorchFbpOperator(TomosipoOperatorMixin, DifferentiableOperator):
    def __init__(self, projector, learn_filter=False, device=None):
        super().__init__(projector=projector, device=device)
        filter = ram_lak(projector.range_shape[-1]).to(self.device)
        filter.requires_grad = True if learn_filter else False
        self.parameters = {'filter': TorchParameter(filter, device=filter.device)}

    @property
    def implements_batching(self):
        return True

    def __call__(self, proj_data):
        proj_data = self._preprocess(proj_data)
        rec = fbp(self.projector, proj_data, filter=self.parameters['filter'].value)
        rec = self._postprocess(rec)
        return rec


class PyTorchFdkOperator(TomosipoOperatorMixin, DifferentiableOperator):
    def __init__(self, projector, learn_filter=False, device=None):
        super().__init__(projector=projector, device=device)
        filter = ram_lak(2* projector.range_shape[-1]).to(self.device)
        filter.requires_grad = True if learn_filter else False
        self.parameters = {'filter': TorchParameter(filter, device=filter.device)}

    @property
    def implements_batching(self):
        return True

    def __call__(self, proj_data):
        proj_data = self._preprocess(proj_data)
        rec = fdk(self.projector, proj_data, filter=self.parameters['filter'].value, padded=True)
        rec = self._postprocess(rec)
        return rec


class PyTorchSirtOperator(TomosipoOperatorMixin, DifferentiableOperator):
    def __init__(self, projector, n_iter, device=None):
        super().__init__(projector=projector, device=device)
        self.parameters = {
            'n_iter': TorchParameter(n_iter, device=device, dtype=torch.float32)
        }

    @property
    def implements_batching(self):
        return True

    def __call__(self, proj_data):
        proj_data = self._preprocess(proj_data)
        num_its = round(self.parameters['n_iter'].value.item())
        rec = sirt(self.projector, proj_data, num_its)
        rec = self._postprocess(rec)
        return rec


class TvMinOperator(TomosipoOperatorMixin, DifferentiableOperator):
    def __init__(self, projector, lam, n_iter=100, device=None):
        super().__init__(projector=projector, device=device)
        if isinstance(lam, torch.Tensor):
            lam = lam.detach().clone().to(self.device).requires_grad_()
        else:
            lam = torch.tensor(lam, device=self.device, requires_grad=True)
        self.parameters = {'lam': TorchParameter(lam, device=device)}
        self.n_iter = n_iter

    @property
    def implements_batching(self):
        return True

    def __call__(self, proj_data):
        proj_data = self._preprocess(proj_data)
        if self.is_2d:
            rec = tv_min2d(self.projector, proj_data,
                               lam=self.parameters['lam'].value,
                               num_iterations=self.n_iter)
        else:
            rec = tv_min3d(self.projector, proj_data,
                               lam=self.parameters['lam'].value,
                               num_iterations=self.n_iter)
        rec = self._postprocess(rec)
        return rec


class ShiftRotationAxisOperator(DifferentiableOperator):
    implements_batching = False

    def __init__(self, method='fourier', device=None):
        super().__init__(device=device)
        self.parameters = {'shift': TorchParameter(0.0, device=device)}
        assert method.lower() in ['fourier', 'grid_sample']
        self.method = method.lower()

    def __call__(self, input):
        input = input.to(self.device)  # [n_rows, n_images, n_cols]
        if self.method == 'fourier':
            fft = torch.fft.fft(input, dim=-1)
            freqs = torch.fft.fftfreq(input.shape[-1], device=input.device)
            shift = self.parameters['shift'].value
            # Apply phase shift
            fft *= torch.exp(-2j * torch.pi * shift * freqs)
            return torch.fft.ifft(fft).real
        elif self.method == 'grid_sample':
            n_rows, n_images, n_cols = input.shape
            shift_normalized = self.parameters['shift'].value / (n_cols/2)
            transform_matrix = torch.tensor([
                [1, 0, 0],
                [0, 1, 0]
            ], dtype=shift_normalized.dtype).repeat([n_images, 1, 1])
            transform_matrix[:, 0, 2] = shift_normalized
            return _apply_affine_transform(input, transform_matrix)


def _apply_affine_transform(tensor, transform_matrix):
    """Apply affine transform encoded by a matrix to a sinogram in NCHW layout."""
    n_rows, n_images, n_cols = tensor.shape
    grid = F.affine_grid(transform_matrix, (n_images, 1, n_rows, n_cols),
                         align_corners=False).to(tensor.device)
    # Input is expected in (N, C, H, W) format
    shifted_images = F.grid_sample(tensor.permute([1, 0, 2]).unsqueeze(dim=1),
                                   grid, align_corners=False, mode='bicubic',
                                   padding_mode='reflection')
    return shifted_images.squeeze(dim=1).permute([1, 0, 2])


def pad_filter(h):
    new_size = list(h.shape)
    new_size[-1] *= 2
    out = h.new_zeros(size=new_size)
    h_width = h.shape[-1]
    out[..., :h_width//2] = h[..., :h_width//2]
    out[..., -h_width//2:] = h[..., -h_width//2:]
    return out


def ram_lak(n):
    # Computes a Ram-Lak filter optimized wrt discretization bias introduced if
    # a naive ramp function is used to filter to the FFT of the projections.
    # For details, see section 3.3.3 in Kak & Staley, "Principles of
    # Computerized Tomographic Imaging", SIAM, 2001.
    filter = torch.zeros(n)
    filter[0] = 0.25
    # even indices are zero
    # for odd indices j, filter[j] equals
    #   -1 / (pi * j) ** 2,          when 2 * j <= n
    #   -1 / (pi * (n - j)) ** 2,    when 2 * j >  n (i.e. circular shift)
    odd_indices = torch.arange(1, n, 2)
    cond = 2 * odd_indices > n
    odd_indices[cond] = n - odd_indices[cond]
    filter[1::2] = -1 / (np.pi * odd_indices) ** 2
    return rfft(filter).real


def filter_sino(sino, filter=None, padded=False):
    """Filter sinogram for use in FBP.

    :param sino: `torch.tensor`
        A three-dimensional sinogram tensor [height x num_angles x width].
    :param filter: `torch.tensor` (optional)
        Filter to apply in Fourier space. This should be as wide as the
        sinogram `y` and can be a 2D tensor if different filters are specified
        for each projection angle or 1D if the same filter should be applied
        everywhere. If not specified, the ram-lak filter is used.
    :param padded: `bool`
        Zero-pad the sinogram before filtering. Default is false.
    :returns:
        A sinogram filtered with the provided filter.
    :rtype: `torch.tensor`

    """
    original_width = sino.shape[-1]
    fft_length = original_width
    if padded:
        fft_length *= 2
    sino_fft = rfft(sino, fft_length)
    if filter is None:
        filter = ram_lak(fft_length).type(sino.dtype).to(sino.device)
    y_filtered = irfft(sino_fft * filter, fft_length)
    if padded:
        y_filtered = y_filtered[..., :original_width].contiguous()

    return y_filtered


def fbp(A, y, padded=False, filter=None, overwrite_y=False):
    if overwrite_y:
        y_filtered = y
    else:
        y_filtered = torch.empty_like(y)

    y_filtered = filter_sino(y, filter, padded)
    rec = A.T(y_filtered)

    # Scale result to make sure that fbp(A, A(x)) == x holds at least
    # to some approximation. In limited experiments, this is true for
    # this version of FBP up to 1%.
    # *Note*: For some reason, we do not have to scale with respect to
    # the pixel dimension that is orthogonal to the rotation axis (`u`
    # or horizontal pixel dimension). Hence, we only scale with the
    # other pixel dimension (`v` or vertical pixel dimension).
    vg, pg = A.astra_compat_vg, A.astra_compat_pg

    pixel_height = (pg.det_size[0] / pg.det_shape[0])
    voxel_volume = np.prod(np.array(vg.size / np.array(vg.shape)))
    scaling = (np.pi / pg.num_angles) * pixel_height / voxel_volume

    rec *= scaling
    return rec


def grad_2D(x):
    weight = x.new_zeros(2, 1, 3, 3)
    weight[0, 0] = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    weight[1, 0] = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    out = torch.conv2d(x, weight, padding=1)
    return out


def grad_2D_T(y):
    weight = y.new_zeros(2, 1, 3, 3)
    weight[0, 0] = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    weight[1, 0] = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    out = torch.conv_transpose2d(y, weight, padding=1)
    return out


def grad_3D(x):
    weight = x.new_zeros(3, 1, 3, 3, 3)
    weight[0, 0] = torch.tensor([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])
    weight[1, 0] = torch.tensor([
        [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])
    weight[2, 0] = torch.tensor([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])
    weight = weight.to(x.device)
    if x.ndim != 4:
        print(x.shape)
        raise Exception("Wrong dimensions for x")
    # x is assumed to be 4 dimensional: Batch + 3 for volume
    x = x.unsqueeze(0) # Add channel dimension
    out = torch.conv3d(x, weight, padding=1)
    return out[:, :, :, :, :]


def grad_3D_T(y):
    weight = y.new_zeros(3, 1, 3, 3, 3)
    weight[0, 0] = torch.tensor([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])
    weight[1, 0] = torch.tensor([
        [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])
    weight[2, 0] = torch.tensor([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])
    weight = weight.to(y.device)
    out = torch.conv_transpose3d(y, weight, padding=1)
    return out[:, 0, ...]      # Remove channel dimension


def operator_norm(A, num_iter=10, device='cpu'):
    x = torch.randn(size=(1, *A.domain_shape), device=device)
    for i in range(num_iter):
        x = A.T(A(x))
        x /= torch.norm(x)  # L2 vector-norm

    norm_ATA = (torch.norm(A.T(A(x))) / torch.norm(x)).item()
    return math.sqrt(norm_ATA)


def operator_norm_plus_grad(A, num_iter=10, device='cpu'):
    x = torch.randn(size=(1, *A.domain_shape), device=device)
    operator_norm_estimate = 0.0
    for i in range(num_iter):
        y_A = A(x)
        y_TV = grad_2D(x)
        x_new = A.T(y_A) + grad_2D_T(y_TV)
        operator_norm_estimate = torch.norm(x_new) / torch.norm(x)
        x = x_new / torch.norm(x_new)

    norm_ATA = operator_norm_estimate.item()
    return math.sqrt(norm_ATA)


def operator_norm_plus_grad3D(A, num_iter=50, device='cpu'):
    x = torch.randn(size=(1, *A.domain_shape), device=device)
    operator_norm_estimate = 0.0
    #print("Estimating 3D operator norm...")
    for i in range(num_iter):
        y_A = A(x)
        y_TV = grad_3D(x)
        grad_T_y_TV = grad_3D_T(y_TV)
        x_new = A.T(y_A) + grad_3D_T(y_TV)
        operator_norm_estimate = torch.norm(x_new) / torch.norm(x)
        x = x_new / torch.norm(x_new)

    norm_ATA = operator_norm_estimate.item()
    return math.sqrt(norm_ATA)


def magnitude(z):
    return torch.sqrt(z[:, 0:1] ** 2 + z[:, 1:2] ** 2)


def clip(z, lamb):
    return z * torch.clamp(lamb / (magnitude(z) + 1e-5), min=None, max=1.0)


def tv_min3d(A, y, lam, num_iterations=50, L=None, non_negativity=False):
    device = y.device

    scale = operator_norm(A, device=device)
    S = tomosipo.scale(1 / scale, pos=A.volume_geometry.pos)
    A = tomosipo.operator(S * A.volume_geometry, S * A.projection_geometry.to_vec())
    A = AutogradedTomosipoOperator(A)  # TODO: Make tomo functions independent from operators
    y = y / scale

    if L is None:
        L = operator_norm_plus_grad3D(A, num_iter=30, device=device)
    t = 1.0 / L
    s = 1.0 / L
    theta = 1

    u = torch.zeros(size=(1, *A.domain_shape), device=device)
    p = torch.zeros(A.range_shape, device=device)
    q = grad_3D(u)                  # contains zeros (and has correct shape)
    u_avg = torch.clone(u)

    #print("Starting TV iteration...")
    for n in range(num_iterations):
        p = (p + s * (A(u_avg) - y)) / (1 + s)
        q = clip(q + s * grad_3D(u_avg), lam)
        u_new = u - (t * A.T(p) + t * grad_3D_T(q))
        if non_negativity:
            u_new = torch.clamp(u_new, min=0.0, max=None)
        u_avg = u_new + theta * (u_new - u)
        u = u_new

    return u.squeeze(1) # Get rid of channel dimension


def tv_min2d(A, y, lam, num_iterations=100, L=None, non_negativity=False):
    dev = y.device
    n_batch = y.shape[0]

    scale = operator_norm(A, device=dev)
    S = tomosipo.scale(1 / scale, pos=A.volume_geometry.pos)
    A = tomosipo.operator(S * A.volume_geometry, S * A.projection_geometry.to_vec())
    A = AutogradedTomosipoOperator(A)  # TODO: Make tomo functions independent from operators
    y = y / scale

    if L is None:
        L = operator_norm_plus_grad(A, num_iter=100, device=dev)
    t = 1.0 / L
    s = 1.0 / L
    theta = 1

    u = torch.zeros(size=(n_batch, *A.domain_shape), device=dev)
    p = torch.zeros(size=(n_batch, *A.range_shape), device=dev)
    q = grad_2D(u)                  # contains zeros (and has correct shape)
    u_avg = torch.clone(u)

    for _ in range(num_iterations):
        p = (p + s * (A(u_avg) - y)) / (1 + s)
        q = clip(q + s * grad_2D(u_avg), lam)
        u_new = u - (t * A.T(p) + t * grad_2D_T(q))
        if non_negativity:
            u_new = torch.clamp(u_new, min=0.0, max=None)
        u_avg = u_new + theta * (u_new - u)
        u = u_new

    return u


def sirt(A, y, num_iterations):
    # Compute C
    y_tmp = torch.ones(size=(1, *A.range_shape), device=y.device)
    C = A.T(y_tmp)
    C[C < tomosipo.epsilon] = math.inf
    C.reciprocal_()

    # Compute R
    x_tmp = torch.ones(size=(1, *A.domain_shape), device=y.device)
    R = A(x_tmp)
    R[R < tomosipo.epsilon] = math.inf
    R.reciprocal_()

    n_batch = y.shape[0]
    x_cur = torch.zeros(size=(n_batch, *A.domain_shape), device=y.device)
    for _ in range(num_iterations):
        y_tmp = A(x_cur)
        y_tmp -= y
        y_tmp *= R

        x_tmp = A.T(y_tmp)
        x_tmp *= C
        x_cur -= x_tmp
    return x_cur


def fdk(A, y, padded=False, filter=None, overwrite_y=False):
    vg = A.astra_compat_vg
    pg = A.astra_compat_pg

    voxel_sizes = np.array(vg.size) / np.array(vg.shape)
    voxel_size_var = np.ptp(voxel_sizes)
    if voxel_size_var > tomosipo.epsilon:
        raise ValueError(
            "The voxels in the volume must have the same size in every dimension. "
            f"Found variation of {voxel_size_var:0.2e}."
        )

    det_size_var = np.ptp(pg.det_sizes, axis=0).max()
    if det_size_var > tomosipo.epsilon:
        raise ValueError(
            "The size of the detector is not constant. "
            f"Found variation of {det_size_var:0.2e}."
        )

    if not (isinstance(pg, tomosipo.geometry.ConeGeometry) or isinstance(pg, tomosipo.geometry.ConeVectorGeometry)):
        raise TypeError(
            "The provided operator A must describe a cone beam geometry."
        )

    # Pre-weigh projections by the inverse of the source-to-pixel distance
    y_weighted = fdk_weigh_projections(A, y, overwrite_y)

    # Compute a normal FBP reconstruction
    return fbp(
        A=A,
        y=y_weighted,
        padded=padded,
        filter=filter,
        overwrite_y=True
    )
