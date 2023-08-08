import numpy as np
import torch
import xraylib

from autodiffCT.operator import DifferentiableOperator
from autodiffCT.parameter import TorchParameter
from autodiffCT.tomography.project import TomosipoOperatorMixin

BOLTZMANN_CONSTANT = 1.3806488e-16  #[erg/k]
SPEED_OF_LIGHT = 299792458e+2  #[cm/s]
PLANCK_CONSTANT = 6.58211928e-19  #[keV*s]
HC = 12.3984198 #keV * Angstrom (10^-10)


class PhaseContrastOperator(TomosipoOperatorMixin, DifferentiableOperator):
    def __init__(self, projector,
                 log_beta=-10,
                 log_delta=-7,
                 propg_dist_cm=15.0,
                 pixel_size_cm=1e-4,
                 beam_energy=20.0,
                 device='cpu'):
        super().__init__(projector=projector, device=device)

        self.parameters = {'log_beta': TorchParameter(log_beta, device=device, requires_grad=True),
                           'log_delta': TorchParameter(log_delta, device=device, requires_grad=True),
                           'propg_dist': TorchParameter(propg_dist_cm, device=device, requires_grad=True),
                           'pixel_size': TorchParameter(pixel_size_cm, device=device, requires_grad=True),
                           'beam_en': TorchParameter(beam_energy, device=device, requires_grad=True)}

    @property
    def implements_batching(self):
        return True

    def __call__(self, vol_data):
        vol_data = self._preprocess(vol_data)

        projected_thickness = self.projector(vol_data)

        phase_contrast_imgs = simulate_phase_contrast_images(
                                        projected_thickness,
                                        self.parameters['log_beta'].value,
                                        self.parameters['log_delta'].value,
                                        self.parameters['propg_dist'].value,
                                        self.parameters['pixel_size'].value,
                                        self.parameters['beam_en'].value)

        phase_contrast_imgs = self._postprocess(phase_contrast_imgs)
        return phase_contrast_imgs


class PaganinPhaseRetrievalOperator(DifferentiableOperator):
    def __init__(self, projector,
                 log_beta=-10,
                 log_delta=-7,
                 propg_dist_cm=15.0,#cm
                 pixel_size_cm=1e-4,
                 beam_energy=20.0,#keV
                 device='cpu'):
        r"""
        BLA
        """
        super().__init__(device=device)
        self.projector = projector

        self.parameters = {'log_beta': TorchParameter(log_beta, device=device),
                           'log_delta': TorchParameter(log_delta, device=device),
                           'propg_dist': TorchParameter(propg_dist_cm, device=device),
                           'pixel_size': TorchParameter(pixel_size_cm, device=device),
                           'beam_energy': TorchParameter(beam_energy, device=device)}

    @property
    def implements_batching(self):
        return False

    def __call__(self, phase_contrast_projs):
        phase_contrast_projs = phase_contrast_projs.to(self.device)

        phase_maps = retrieve_phase(phase_contrast_projs,
                                    log_beta=self.parameters['log_beta'].value,
                                    log_delta=self.parameters['log_delta'].value,
                                    propg_dist=self.parameters['propg_dist'].value,
                                    pixel_size=self.parameters['pixel_size'].value,
                                    beam_energy=self.parameters['beam_energy'].value)
        return phase_maps.to(self.device)

    @property
    def output_shape(self, input_dims):
        pass

    @property
    def input_shape(self):
        pass


def simulate_refractive_attenuation(phantom,
                                  compound1='CaCO3',
                                  compound2='(O2)0.209476(N2)0.78084(Ar)0.00934',
                                  beam_energy=20,
                                  density1=0.997,
                                  density2=0.001225):
    r"""
    Simulates phase contrast image based on transport of intensity function as implemented
    in TomoPy.

    Assumes single material and background material, which is also what the
    phase retrieval algorithm assumes.

    Args:
        phantom (torch.tensor): Volume phantom to be imaged as binary tensor for two materials.
        compound1 (str): Molecular formula of material to be imaged. Default=CaCO3 (bone).
        compound2 (str): Molecular formula of background material. Default=Air.
        beam_energy (float): Beam energy in KeV. Default=20 keV.
        density (float): Density of compound1. Default=1.0.
    """
    # Refractive indices of materials
    # The refraction (delta) for phase retrieval is defined as 1 - n,
    # with n the refractive index.
    refraction1 = 1.0 - xraylib.Refractive_Index_Re(E=beam_energy, compound=compound1, density=density1)
    refraction2 = 1.0 - xraylib.Refractive_Index_Re(E=beam_energy, compound=compound2, density=density2)
    refractive_vol = phantom.clone().to(torch.float64)
    refractive_vol *= refraction1
    refractive_vol[refractive_vol == 0] += refraction2

    beta1 = xraylib.Refractive_Index_Im(E=beam_energy, compound=compound1, density=density1)
    #9.8663479e-9 is h * c / 4*pi
    #K = 2*pi/lambda, lambda = h*c/E, so K=2*pi*E/(h c)
    K = beam_energy / (2*9.8663479e-9) # = 4 * pi * E/ (h * c)
    mu1 = beta1 * K * 2.0 / density1

    # h * c = 1.23984198 eV * micro m

    # k = 2pi/lambda, mu = 2k*beta = 4pi beta / lambda.
    # lambda = hc / E, so:  mu = (4pi * beta * E (in keV)) / hc (12.398)
    # This gives mu in 1 / Angstrom units.
    mux = 4 * np.pi * beta1 * beam_energy / HC
    # mu1 is in 1/Angstrom now. We assume all the arguments are in cm,
    #  so multiply by 1e8 so that mu is in 1/cm units
    mux *= 1e8
    mu1 = mux

    mu_vol = phantom.clone()
    mu_vol *= mu1

    beta2 = xraylib.Refractive_Index_Im(E=beam_energy, compound=compound2, density=density2)
    mu2 = beta2 * K * 2.0 / density2

    muy = 4 * np.pi * beta2 * beam_energy / HC
    # mu1 is in 1/Angstrom now. We assume all the arguments are in cm,
    #  so multiply by 1e8
    muy *= 1e8
    mu2 = muy

    mu_vol[mu_vol == 0] += mu2
    return mu_vol, refractive_vol, mu1, mu2, refraction1, refraction2


def wavelength_en(energy):
    r"""
    Args:
        energy (float): Beam energy in keV.

    Returns:
        wavelength in cm
    """
    return 1e-8 * HC / energy


def padding_width(dim, pixel_size_cm, wavelength, propg_dist):
    pad_pix = torch.ceil(np.pi * wavelength * propg_dist / pixel_size_cm ** 2)
    return int((torch.pow(2, torch.ceil(torch.log2(dim + pad_pix))) - dim) * 0.5)


def calc_padding(proj_data, beam_energy, propg_dist, pixel_size):
    dx, dy, dz = proj_data.shape
    wavelength = wavelength_en(beam_energy)
    val = ((proj_data[..., 0] + proj_data[..., -1]) * 0.5).mean()
    py = padding_width(dy, pixel_size, wavelength, propg_dist)
    pz = padding_width(dz, pixel_size, wavelength, propg_dist)
    return py, pz, val


def reciprocal_grid(pixel_size, nx, ny):
    """
    Calculate reciprocal grid.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    nx, ny : int
        Size of the reciprocal grid along x and y axes.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    # Sampling in reciprocal space.
    indx = 2*np.pi*torch.fft.fftfreq(nx, device=pixel_size.device) / pixel_size
    indy = 2*np.pi*torch.fft.fftfreq(ny, device=pixel_size.device) / pixel_size
    indx = indx**2
    indy = indy**2
    # This computes the square of the distances from each
    #  grid point to the origin in reciprocal space.
    # This will be the sampling the grid in Fourier domain.
    return indx[:, None] + indy[None, :]


def paganin_filter_factor(energy, dist, alpha, w2):
    # The equation is changed according to Paganin equation.
    # Alpha represents the ratio of delta/beta.
    return 1 / (1 + (dist * alpha * w2 * wavelength_en(energy)/(4*np.pi)))
    # If alpha were defined as alpha/mu, this would be the equation
    #return 1 / (1 + (dist * alpha * w2))


def retrieve_phase(phase_contrast_data, log_beta=-10, log_delta=-7, propg_dist=15.0,
                   pixel_size=1e-4, beam_energy=20.0, pad=False):
    r"""
    This function is a re-implementation from tomopy.prep.phase.retrieve_phase in PyTorch primitives.
    """
    # To optimize log(delta) instead, we need to undo it here
    delta = torch.exp(log_delta)
    beta = torch.exp(log_beta)

    if pad:
        py, pz, val = calc_padding(phase_contrast_data, pixel_size, propg_dist, beam_energy)
    else:
        py, pz, val = 0, 0, torch.tensor(0.0)

    dx, dy, dz = phase_contrast_data.shape
    w2 = reciprocal_grid(pixel_size, dy + 2 * py, dz + 2 * pz)

    # Filter in Fourier space.
    phase_filter = paganin_filter_factor(beam_energy, propg_dist, delta/beta, w2)

    # In TomoPy, they sample -n to n with steps of 2, thereby missing the 0
    # frequency when n is odd. Also, frequencies are scaled by a factor of 2.
    # We sample from -n/2 to n/2, with 0 included, and do not need normalization.
    #normalized_phase_filter = phase_filter / phase_filter.max()

    if pad:
        prj = torch.full((dx, dy + 2 * py, dz + 2 * pz), val.item(), dtype=torch.float32,
                         device=phase_contrast_data.device)#
        phase_maps = torch.empty_like(phase_contrast_data)

        prj[:, py:dy + py, pz:dz + pz] = phase_contrast_data#
        prj[:,:py,:] = val
        prj[:,-py:,:] = val
        prj[:, :, :pz] = prj[:, :, pz][:, :, None]#
        prj[:, :, -pz:] = prj[:, :, -pz-1][:, :, None]#

        #fproj *= normalized_phase_filter[None,...]
        fproj = torch.fft.fft2(prj) # dim should be correct, i.e., over last 2 dimensions

        fproj *= phase_filter[None,...]

        proj = torch.real(torch.fft.ifft2(fproj)) # dim should be over last 2 dimensions

        proj = proj[:, py:dy + py, pz:dz + pz]
        del prj
    else:
        # dim of FFT should be correct, i.e., over last 2 dimensions
        fproj = torch.fft.fft2(phase_contrast_data)

        #fproj *= normalized_phase_filter[None,...]
        fproj *= phase_filter[None,...]

        proj = torch.real(torch.fft.ifft2(fproj)) # dim should be over last 2 dimensions

    # TomoPy outputs the raw output after the inverse FFT.
    # According to [1] we need -log( . )
    proj = -torch.log(proj)

    #NOTE: IF PULL REQUEST VERSION
    #proj = proj/(4*np.pi/wavelength_en(beam_energy))

    # NOTE: [1] assumes a single material so if we fill the background with air
    # the reconstruction is not perfect.

    # [1]: Paganin D, Mayo SC, Gureyev TE, Miller PR, and Wilkins SW.
    #      Simultaneous phase and amplitude extraction from a single defocused
    #      image of a homogeneous object

    del fproj
    del phase_filter
    return torch.swapaxes(proj, 0, 1)


def simulate_phase_contrast_images(projected_thickness, log_beta, log_delta, propagation_dist, pixel_size, beam_energy):
    r"""
    Args:
        projected thickness (torch.tensor): Projected thickness of object with shape
                                            (detector_size_x, Nr_angles, detector_size_y).
                                            This volume should not have units of length.
        refraction_obj (float): Refractive index (delta), i.e., real part of complex refractive index of wave.
                                Has no units, but should be calculated using cm and grams as units for material.
        mu_obj (float): linear attenuation coefficient (mu), unit should be 1/cm.
        propagation_dist (float): Propagation distance of the wavefront in cm.
        pixel_size (float): Detector pixel size in cm.
    """
    # To optimize log(delta) instead, we need to undo it here
    delta = torch.exp(log_delta)
    beta = torch.exp(log_beta)

    # Input projected thickness has no units, it is summed over the tomosipo volume.
    # Pixel size is in cm
    #projected_thickness = projected_thickness * pixel_size
    #projected_thickness = projected_thickness.to(torch.float64)

    # mu_obj must be in 1/cm units.
    # Here, we implement the left-hand side of eq. 7 of Paganin's paper [1]
    # atten_proj is e^(-mu * T(r_perp)), where r_perp is the plane perpendicular to the optical axis z.
    # Input projected thickness has no units, it is summed over the tomosipo volume.
    # Pixel size is in cm so we have to correct for that too by multiplying by pixel_size
    atten_proj = torch.exp(-4 * np.pi * beta * pixel_size * projected_thickness/wavelength_en(beam_energy))

    # In eq. 7, we apply the laplacian in the plane of r_perp to "e^(-mu * T(r_perp))", i.e., atten_proj.
    # After applying tomosipo projector, the returned projections are of the shape (Px, NAngles, Py).
    # Hence we apply gradients in dimensions 0 and 2, with pixel_size spacing.
    if atten_proj.ndim == 3:
        grad_x, grad_y = torch.gradient(atten_proj, dim=(0, 2))
        grad_x2, = torch.gradient(grad_x, dim=0)
        grad_y2, = torch.gradient(grad_y, dim=2)
    elif atten_proj.ndim == 4:
        phase_ims = torch.empty_like(atten_proj)
        for b_idx in range(atten_proj.shape[0]):
            grad_x, grad_y = torch.gradient(atten_proj[b_idx], dim=(0, 2))
            grad_x2, = torch.gradient(grad_x, dim=0)
            grad_y2, = torch.gradient(grad_y, dim=2)

            laplace = (grad_x2 + grad_y2) / pixel_size**2

            phase_im = atten_proj[b_idx] - (propagation_dist * delta * wavelength_en(beam_energy) * laplace/(4 * beta * np.pi))
            phase_ims[b_idx] = phase_im
        return phase_ims.swapaxes(1,2)

    laplace = (grad_x2 + grad_y2) / pixel_size**2

    # This is an implementation of eq. 7 of Paganin's paper with incident intensity (I_in) assumed to be 1.
    # Note: lapace is now equal to Nabla^2_perp * e^(-mu * T(r_perp)) in the paper.

    phase_im = atten_proj - (propagation_dist * delta * wavelength_en(beam_energy) * laplace/(4 * beta * np.pi))
    # [1]: Paganin D, Mayo SC, Gureyev TE, Miller PR, and Wilkins SW.
    #      Simultaneous phase and amplitude extraction from a single defocused image of a homogeneous object

    # Swap axes so that shape of contrast image is (NAngles, Px, Py)
    if atten_proj.ndim == 3:
        return phase_im.swapaxes(0,1)
