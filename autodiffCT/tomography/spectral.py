import torch
import xraylib
from torch.utils.checkpoint import checkpoint as checkpoint_gradients


def mass_attenuation(energies, compound):
    """
    Total X-ray absorption for a given compound in cm2g.
    Units: KeV.

    """
    attenuations = [xraylib.CS_Total_CP(compound, e.item()) for e in energies]
    return torch.tensor(attenuations, device=energies.device)


def bremsstrahlung(energies, energy_max):
    r"""
    Simple bremstrahlung model (Kramer formula). Emax
    """
    # Kramer:
    spectrum = energy_max / energies - 1
    spectrum[spectrum < 0] = 0
    return spectrum / spectrum.mean()


def scintillator_efficiency(energies, compound, rho, thickness):
    """
    Generate QDE of a detector (scintillator).
    Units: KeV, g/cm3, mm.

    References: 10.3390/jimaging6040018, 10.1118/1.595357, 10.1109/23.682433

    """
    # Attenuation by the photoelectric effect:
    photoelectric = [xraylib.CS_Photo_CP(compound, e.item()) for e in energies]
    photoelectric = torch.tensor(photoelectric, device=energies.device)
    spectrum = 1 - torch.exp(-rho * photoelectric * thickness/10)
    spectrum *= energies  # spectral response is assumed to be proportional to E
    return spectrum / spectrum.mean()


def total_transmission(energies, compound, rho, thickness):
    """
    Compute fraction of x-rays transmitted through the filter.
    Units: KeV, g/cm3, mm.

    """
    return torch.exp(-rho * mass_attenuation(energies, compound) * thickness/10)


def effective_spectrum(energies, acceleration_voltage=90.0, filter=None, detector=None):
    """
    Generate an effective spectrum of a CT scanner.
    Units: keV, kV.

    """
    if filter is None:
        filter = {'material': 'Al', 'density': 2.7, 'thickness': 2.5}
    if detector is None:
        detector = {'material': 'CsI', 'density': 4.51, 'thickness': 0.5}
    # Tube:
    spectrum = bremsstrahlung(energies, acceleration_voltage)
    # Filter:
    spectrum *= total_transmission(energies, filter['material'],
                                   filter['density'], filter['thickness'])
    # Detector:
    spectrum *= scintillator_efficiency(energies, detector['material'],
                                        detector['density'], detector['thickness'])
    return spectrum / spectrum.mean()


def bilinear_attenuation(energies, Z_rel):
    """
    Bilinear parametrization of X-ray absorption spectra of Martinez et al.
    Units: keV, unitless, g/cm3.

    Reference: 10.1016/j.apradiso.2015.09.014

    """
    reference_mu = mass_attenuation(energies, 'H2O')

    def alpha(energies):
        A1 = 12179.0
        p = 2.8
        return 1 / (1 + A1*torch.pow(energies, -p))

    return reference_mu * (alpha(energies)*(1 - Z_rel**3.21) + Z_rel**3.21)


def spectral_projection(projector, energies, effective_spectrum,
                        density_vols, compounds=None, Z_rels=None):
    """
    Simulate projections taking integration over spectral domain into account.

    """
    if compounds is not None and Z_rels is None:
        attenuations = torch.stack([mass_attenuation(energies, x) for x in compounds])
    elif Z_rels is not None and compounds is None:
        attenuations = torch.stack([bilinear_attenuation(energies, x) for x in Z_rels])
    else:
        raise ValueError('One of `compounds` or `Z_rels` should be specified.')

    density_projs = torch.stack([projector(x) for x in density_vols])

    def counts_E(intensity_E, density_projs, attenuations_E):
        # Beer-Lambert law for multiple materials at a given photon energy
        absorbance_E = torch.tensordot(density_projs, attenuations_E, dims=[[0], [0]])
        return intensity_E * torch.exp(-absorbance_E)

    if any([effective_spectrum.requires_grad, density_projs.requires_grad,
            attenuations.requires_grad]):
        # We probably do pipeline optimization (i.e. multiple function calls),
        # so compilation is justified
        counts_E_compiled = torch.jit.script(counts_E)
        # We are typically limited by memory if backward pass needs to be
        # calculated. Use gradient checkpointing to trade compute for memory
        counts_E = lambda *x: checkpoint_gradients(counts_E_compiled, *x)

    simulated_counts = torch.zeros_like(density_projs[0])
    for intensity_E, attenuations_E in zip(effective_spectrum, attenuations.T):
        simulated_counts += counts_E(intensity_E, density_projs, attenuations_E)
    simulated_counts /= effective_spectrum.sum()
    simulated_ints = -torch.log(simulated_counts)

    return simulated_ints
