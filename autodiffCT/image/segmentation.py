import torch
import torch.nn as nn
from skimage.filters import threshold_multiotsu

from autodiffCT.operator import DifferentiableOperator
from autodiffCT.parameter import TorchParameter


class ThresholdOperator(DifferentiableOperator):
    def __init__(self, thresholds=None, init_method='otsu', n_classes=2,
                 bins=128, device='cpu', re_init_at_call=True,
                 gamma=1e4, split_classes=False, normalize=False):
        super().__init__(device=device)
        self.supported_methods = ["otsu"]
        self.init_method = init_method
        if self.init_method not in self.supported_methods:
            raise Exception(f"Unknown method {init_method} passed for initializing thresholds")

        if not isinstance(n_classes, int):
            raise Exception("n_classes has to be an integer")
        self.n_classes = n_classes
        self.bins = bins
        self.gamma = gamma
        self.normalize = normalize

        self.parameters = {}
        if thresholds is not None:
            self.parameters['thresholds'] = TorchParameter(thresholds, device=self.device)
        else:
            self.parameters['thresholds'] = None
        self.split_classes = split_classes
        self.re_init_at_call = re_init_at_call

    @property
    def implements_batching(self):
        return False

    def normalize_volume(self, volume):
        t = volume.min()
        volume = volume - t
        s = volume.max()
        volume = (1/s) * volume
        return volume

    def initialize_threshold(self, volume):
        if self.init_method == 'otsu':
            thresholds = threshold_multiotsu(volume.cpu().detach().numpy(),
                                             classes=self.n_classes,
                                             nbins=self.bins)
        self.parameters['thresholds'] = TorchParameter(thresholds, device=self.device)

    def __call__(self, volume):
        if self.normalize:
            volume = self.normalize_volume(volume)
        if self.re_init_at_call:
            self.initialize_threshold(volume)
        if not self.split_classes:
            segmented_volume = soft_thresholds(volume.to(self.device),
                                               self.parameters['thresholds'].value,
                                               gamma=self.gamma,
                                               device=self.device)
            return segmented_volume.to(self.device)
        else:
            segm_masks = soft_thresholds_split(volume.to(self.device),
                                                           self.parameters['thresholds'].value,
                                                           gamma=self.gamma,
                                                           device=self.device)
            segm_masks = segm_masks.to(self.device)
            return segm_masks

    @property
    def output_shape(self, input_dims):
        return input_dims

    @property
    def input_shape(self):
        return (float('inf'),)


class SegmentationMaskOperator(DifferentiableOperator):
    def __init__(self, thresholds=None, init_method='otsu', n_classes=2,
                 bins=128, device='cpu', re_init_at_call=True,
                 gamma=1e4):
        super().__init__(device=device)
        self.supported_methods = ["otsu"]
        self.init_method = init_method
        if self.init_method not in self.supported_methods:
            raise Exception(f"Unknown method {init_method} passed for initializing thresholds")

        if not isinstance(n_classes, int):
            raise Exception("n_classes has to be an integer")
        self.n_classes = n_classes
        self.bins = bins
        self.gamma = gamma

        self.parameters = {}
        if thresholds is not None:
            self.parameters['thresholds'] = TorchParameter(thresholds, device=self.device)
        else:
            self.parameters['thresholds'] = None
        self.re_init_at_call = re_init_at_call

    @property
    def implements_batching(self):
        return False

    def initialize_threshold(self, volume):
        if self.init_method == 'otsu':
            thresholds = threshold_multiotsu(volume.cpu().detach().numpy(),
                                             classes=self.n_classes,
                                             nbins=self.bins)
        self.parameters['thresholds'] = TorchParameter(thresholds, device=self.device)

    def __call__(self, volume):
        if self.re_init_at_call:
            self.initialize_threshold(volume)
        segmented_volume = soft_thresholds(volume.to(self.device),
                                           self.parameters['thresholds'].value,
                                           gamma=self.gamma,
                                           device=self.device)
        return segmented_volume.to(self.device) * volume.squeeze(0)

    def get_output_dimensions(self, input_dims):
        return input_dims

    def required_input_dimensions(self):
        return (float('inf'),)


class MultiOtsuOperator(DifferentiableOperator):
    def __init__(self, n_classes=2, bins=128, device='cpu', gamma=1e4, downsample=None):
        super().__init__(device=device)
        self.parameters = {}
        self.n_classes = n_classes
        self.gamma = gamma

        if not isinstance(n_classes, int):
            raise Exception("n_classes has to be an integer")
        if n_classes < 2 or n_classes > 4:
            raise Exception("Supports only 2,3,4 classes")

        self.bins = bins
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample_layer = torch.nn.AvgPool3d(self.downsample)

    @property
    def implements_batching(self):
        return False

    def __call__(self, volume):
        if self.downsample is None:
            if self.n_classes == 2:
                segm, thresholds = otsu2_torch(volume,
                                               nbins=self.bins,
                                               device=self.device,
                                               gamma=self.gamma)
            elif self.n_classes == 3:
                segm, thresholds = otsu3_torch(volume,
                                               nbins=self.bins,
                                               device=self.device,
                                               gamma=self.gamma)
            elif self.n_classes == 4:
                segm, thresholds = otsu4_torch(volume,
                                               nbins=self.bins,
                                               device=self.device,
                                               gamma=self.gamma)
            return segm
        else:
            # Otsu does not implement batching, add the dimension for torch
            volume = volume.unsqueeze(0)
            small_volume = self.downsample_layer(volume)
            with torch.no_grad():
                if self.n_classes == 2:
                    vol, thresholds = otsu2_torch(small_volume,
                                                  nbins=self.bins,
                                                  device=self.device,
                                                  gamma=self.gamma,
                                                  apply_thresholds=False)
                elif self.n_classes == 3:
                    vol, thresholds = otsu3_torch(small_volume,
                                                  nbins=self.bins,
                                                  device=self.device,
                                                  gamma=self.gamma,
                                                  apply_thresholds=False)
                elif self.n_classes == 4:
                    vol, thresholds = otsu4_torch(small_volume,
                                                  nbins=self.bins,
                                                  device=self.device,
                                                  gamma=self.gamma,
                                                  apply_thresholds=False)
            vol = torch.nn.functional.upsample(vol.unsqueeze(0), size=volume.size()[1:])
            segmented_volume = soft_thresholds(vol.squeeze(0).squeeze(0),
                                               thresholds, gamma=self.gamma)
            return segmented_volume.to(self.device)

    @property
    def output_shape(self, input_dims):
        return input_dims

    @property
    def input_shape(self):
        return (float('inf'),)


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = (float(min) + self.delta * (torch.arange(bins).float() + 0.5))

    def forward(self, x):
        x = x.cpu().flatten()
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x


def soft_thresholds(vol, thresholds, gamma=100.0, b=1.0, device='cpu'):
    r"""
    Returns segmented volume where voxels < thresholds[0] are 0.0,
        thresholds[0] < voxels < thresholds[1] are 1.0,
        between thresholds[1] and thresholds[2] is 2.0 etc.
    """
    vol1 = torch.zeros(size=vol.shape, device=device)
    vol = vol.to(device)

    # Ensure 0 to 1 bounds of values, scale thresholds accordingly
    mi = vol.min()
    vol = vol - mi
    thresholds = thresholds - mi
    ma = vol.max()
    vol = (1/ma) * vol
    thresholds = (1/ma) * thresholds

    for i, t in enumerate(thresholds):
        # 1 + 0.5* is to scale tanh to [0,1] range
        #tanh centered around b, then times gamma
        vol1 += 0.5 * (1 + torch.tanh(gamma*((vol / t)-b)))
    return vol1


def soft_thresholds_split(vol, thresholds, gamma=100.0, b=1.0, device='cpu'):
    r"""
    Returns separate binary masks selecting voxels for each class.
        First mask is where voxels < thresholds[0],
        second mask is where thresholds[0] < voxels < thresholds[1],
        next is between thresholds[1] and thresholds[2] etc..
    """
    vol1 = torch.zeros(size=(len(thresholds)+1,) + vol.shape, device=device)
    vol = vol.to(device)
    thrs = thresholds.sort()[0] # Ensure ascending order of thresholds

    # Ensure 0 to 1 bounds of values, scale thresholds accordingly
    mi = vol.min()
    vol = vol - mi
    thrs = thrs - mi
    ma = vol.max()
    vol = (1/ma) * vol
    thrs = (1/ma) * thrs

    # Initialize lower bin as every voxel
    vol1[0] = vol1[0] + 1.0
    for i, t in enumerate(thrs):
        # everything larger than t is set to 1 in a soft/differentiable manner
        # 1 + 0.5* is to scale tanh to [0,1] range
        #tanh centered around b, then times gamma
        vol1[i+1] = 0.5 * (1 + torch.tanh(gamma*((vol / t)-b)))

    # This makes sure that the masks are a partition.
    for i in reversed(range(len(thrs))):
        for j in range(i + 1, len(thrs)+1):
            vol1[i] = vol1[i] - vol1[j]
    return vol1


def apply_thresholds(vol, thresholds):
    segm_vol = torch.zeros(size=vol.shape)
    for i, t in enumerate(thresholds):
        segm_vol[vol > t] = i + 1
    return segm_vol


def apply_thresholds_split(vol, thresholds, device='cpu'):
    r"""
    Returns separate binary masks selecting voxels for each class.
        First mask is where voxels < thresholds[0],
        second mask is where thresholds[0] < voxels < thresholds[1],
        next is between thresholds[1] and thresholds[2] etc..
    """
    vol1 = torch.zeros(size=(len(thresholds)+1,) + vol.shape, device=device, dtype=torch.int32)
    vol = vol.to(device)
    thrs = thresholds.sort()[0] # Ensure ascending order of thresholds

    # Initialize lower bin as every voxel
    vol1[0] = vol1[0] + 1
    for i, t in enumerate(thrs):
        # everything larger than t is set to 1
        vol1[i+1][vol > t] = 1

    # This makes sure that the masks are a partition.
    for i in reversed(range(len(thrs))):
        for j in range(i + 1, len(thrs)+1):
            vol1[i] = vol1[i] - vol1[j]
    return vol1


def otsu2_torch(volume, nbins=256, device='cpu', gamma=100.0, apply_thresholds=True):
    t = volume.min()
    vol1 = volume.clone().to(device)
    vol1 = volume - t
    s = vol1.max()
    vol2 = vol1.clone()
    vol2 = (1/s) * vol1

    softhist = SoftHistogram(bins=nbins, min=0, max=1, sigma=100000)
    hist = softhist(vol2)
    bin_centers = softhist.centers
    N = hist.sum()

    nvalues = torch.count_nonzero(hist)
    if nvalues < 2:
        raise ValueError("The input volume has less values than number of classes.")

    MT = 0
    for k in range(nbins):
        pk = hist[k] / N
        MT += k * pk

    optimalT1 = None
    W0K = 0
    M0K = 0
    maxBetweenVar = 0

    for t1 in range(nbins):
        p1 = hist[t1] / N
        W0K += p1
        M0K += t1 * p1
        M0 = M0K / W0K

        W1K = 1.0 - W0K
        M1K = MT - M0K

        if W1K <= 0:
            break

        M1 = M1K / W1K
        currVarB = W0K * (M0 - MT) * (M0 - MT) + W1K * (M1 - MT) * (M1 - MT)

        if maxBetweenVar < currVarB:
            maxBetweenVar = currVarB
            optimalT1 = t1

    thresholds = bin_centers[torch.tensor([optimalT1])]
    if not apply_thresholds:
        return vol2, thresholds

    segmented_volume = soft_thresholds(vol2, thresholds, gamma=gamma)
    return segmented_volume.to(device), thresholds


def otsu3_torch(volume, nbins=256, device='cpu', gamma=100.0, apply_thresholds=True):
    t = volume.min()
    vol1 = volume.clone().to(device)
    vol1 = volume - t
    s = vol1.max()
    vol2 = vol1.clone()
    vol2 = (1/s) * vol1

    softhist = SoftHistogram(bins=nbins, min=0, max=1, sigma=100000)
    hist = softhist(vol2)
    bin_centers = softhist.centers
    N = hist.sum()

    nvalues = torch.count_nonzero(hist)
    if nvalues < 3:
        raise ValueError("The input volume has less values than number of classes.")

    MT = 0
    for k in range(nbins):
        pk = hist[k] / N
        MT += k * pk

    optimalT1 = None
    optimalT2 = None
    W0K = 0
    M0K = 0
    maxBetweenVar = 0

    for t1 in range(nbins):
        p1 = hist[t1] / N
        W0K += p1
        M0K += t1 * p1
        M0 = M0K / W0K

        W1K = 0
        M1K = 0
        for t2 in range(t1+1, nbins):
            p2 = hist[t2] / N
            W1K += p2
            M1K += t2 * p2
            M1 = M1K / W1K

            W2K = 1.0 - (W0K + W1K)
            M2K = MT - (M0K + M1K)

            if W2K <= 0:
                break

            M2 = M2K / W2K
            currVarB = W0K * (M0 - MT) * (M0 - MT) + W1K * (M1 - MT) * (M1 - MT) + W2K * (M2 - MT) * (M2 - MT)

            if maxBetweenVar < currVarB:
                maxBetweenVar = currVarB
                optimalT1 = t1
                optimalT2 = t2

    thresholds = bin_centers[torch.tensor([optimalT1, optimalT2])]
    if not apply_thresholds:
        return vol2, thresholds

    segmented_volume = soft_thresholds(vol2, thresholds, gamma=gamma)
    return segmented_volume.to(device), thresholds


def otsu4_torch(volume, nbins=256, device='cpu', gamma=100.0, apply_thresholds=True):
    t = volume.min()
    vol1 = volume.clone().to(device)
    vol1 = volume - t
    s = vol1.max()
    vol2 = vol1.clone()
    vol2 = (1/s) * vol1

    softhist = SoftHistogram(bins=nbins, min=0, max=1, sigma=100000)
    hist = softhist(vol2)
    bin_centers = softhist.centers
    N = hist.sum()

    nvalues = torch.count_nonzero(hist)
    if nvalues < 4:
        raise ValueError("The input volume has less values than number of classes.")

    MT = 0
    for k in range(nbins):
        pk = hist[k] / N
        MT += k * pk

    optimalT1 = None
    optimalT2 = None
    optimalT3 = None
    W0K = 0
    M0K = 0
    maxBetweenVar = 0

    for t1 in range(nbins):
        p1 = hist[t1] / N
        W0K += p1
        M0K += t1 * p1
        M0 = M0K / W0K

        W1K = 0
        M1K = 0
        for t2 in range(t1+1, nbins):
            p2 = hist[t2] / N
            W1K += p2
            M1K += t2 * p2
            M1 = M1K / W1K

            W2K = 0
            M2K = 0
            for t3 in range(t2+1, nbins):
                p3 = hist[t3] / N
                W2K += p3
                M2K += t3 * p3
                M2 = M2K / W2K

                W3K = 1 - (W0K + W1K + W2K)
                M3K = MT - (M0K + M1K + M2K)
                if W3K <= 0:
                    break

                M3 = M3K / W3K
                currVarB = W0K * (M0 - MT) * (M0 - MT) + W1K * (M1 - MT) * (M1 - MT) + W2K * (M2 - MT) * (M2 - MT)+ W3K * (M3 - MT) * (M3 - MT)

                if maxBetweenVar < currVarB:
                    maxBetweenVar = currVarB
                    optimalT1 = t1
                    optimalT2 = t2
                    optimalT3 = t3

    thresholds = bin_centers[torch.tensor([optimalT1, optimalT2, optimalT3])]
    if not apply_thresholds:
        return vol2, thresholds

    segmented_volume = soft_thresholds(vol2, thresholds, gamma=gamma)
    return segmented_volume.to(device), thresholds
