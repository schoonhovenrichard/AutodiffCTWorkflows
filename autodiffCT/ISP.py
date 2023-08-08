import torch
from torch.utils.checkpoint import checkpoint as checkpoint_gradients
from autodiffCT.image.segmentation import soft_thresholds_split, apply_thresholds_split, SegmentationMaskOperator
from itertools import combinations
from autodiffCT.tomography.reconstruction import fbp, fdk
from skimage.filters import threshold_multiotsu
from tqdm import trange

import numpy as np


def spectral_projection_atten(projector, energies, effective_spectrum, attenuations, density_vols):
    """
    Simulate projections taking integration over spectral domain into account.

    """
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


def torch_minimize(params, objective_func, iterations=100, lr=0.01, func_args=[], constraints=None):
    if isinstance(params, list):
        opt = torch.optim.SGD(params, lr=lr)
    else:
        opt = torch.optim.SGD([params], lr=lr)

    final_loss = 0.0
    for it in range(iterations):
        loss = objective_func(params, *func_args)
        loss_diff = final_loss
        final_loss = loss.item()
        loss_diff = abs(loss_diff - final_loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if constraints is not None:
            with torch.no_grad():
                if isinstance(params, list):
                    for par in params:
                        if isinstance(par, dict):
                            par['params'][0].data = par['params'][0].data.clamp(constraints[0], constraints[1])
                            par['params'][0].requires_grad = True
                        else:
                            par.data = par.data.clamp(constraints[0], constraints[1])
                            par.requires_grad = True
                else:
                    params.data = params.data.clamp(constraints[0], constraints[1])
                    params.requires_grad = True
        if loss_diff < 1e-11:
            print("Stopped at iteration:", it)
            break


def ISP_objective_func(y_meas, A, energies, spectrum, attenuations, material_volumes, losses=None):
    y_simulated = spectral_projection_atten(A,
                                            energies,
                                            spectrum,
                                            attenuations,
                                            material_volumes)
    loss = torch.nn.functional.mse_loss(y_simulated, y_meas.unsqueeze(0))
    if losses is not None:#Optionally track losses:
        losses.append(loss.item())
    return loss


def ISP_thresholds_objective(thresholds, y_meas, A, R_i, energies, spectrum, attenuations, gamma=100.0, losses=None):
    material_volumes = soft_thresholds_split(R_i, thresholds, gamma=gamma, device=thresholds.device)
    return ISP_objective_func(y_meas, A, energies, spectrum, attenuations, material_volumes, losses=losses)


def bruteforce_thresholds_argmin(thresholds, y_meas, A, R_i, energies, spectrum, attenuations, bins=64, losses=None):
    with torch.no_grad():
        best_loss = ISP_thresholds_objective(thresholds, y_meas, A, R_i, energies, spectrum, attenuations, losses=losses)
        best_thresholds = thresholds.clone()

        delta = float(R_i.max() - R_i.min()) / float(bins)
        centers = list(float(R_i.min()) + delta * (torch.arange(bins).float() + 0.5))
        centers.sort()# Sort centers
        N = thresholds.shape[0]

        for combo in combinations(centers, N):  # 2 for pairs, 3 for triplets, etc, we use N
            new_thres = torch.tensor(combo, device=thresholds.device)
            loss = ISP_thresholds_objective(new_thres, y_meas, A, R_i, energies, spectrum, attenuations, losses=losses)
            if loss < best_loss:
                best_loss = loss
                best_thresholds = new_thres.clone()
        return best_thresholds


def ISP_atten_objective(args, material_volumes, y_meas, A, energies, losses=None):
    attenuations = args[0]['params'][0]
    spectrum =  args[1]['params'][0]
    loss = ISP_objective_func(y_meas, A, energies, spectrum, attenuations, material_volumes, losses=losses)
    return loss


def ISP(y_meas, A, N, energies, spectrum, cone=False, attenuations=None, steps=1, losses=None, thr_bins=64, lmin_steps=100, lrs=[1e-1, 1e-1]):
    r"""
    y_meas is the input measures sinogram (which has beamhardening artifacts), A is the
    tomosipo projector operator, N is the number of materials, spectrum is the beam spectrum
    NOTE: This assumes y_meas is flat-fielded, and -log(
    """
    # Nr of energy bins
    E_m = spectrum.shape[0]
    device = spectrum.device

    # Initial reconstrunction
    with torch.no_grad():
        if cone:
            R0 = fdk(A, y_meas.unsqueeze(0), padded=True).to(device)
        else:
            R0 = fbp(A, y_meas.unsqueeze(0), padded=True).to(device)

    # Compute initial reasonable thresholds
    with torch.no_grad():
        objreg_op = SegmentationMaskOperator(bins=64, device=device, n_classes=2,
                                             gamma=1000.0, re_init_at_call=True)
        thresholds = threshold_multiotsu(objreg_op(R0).cpu().detach().numpy(),
                                         classes=N,
                                         nbins=128)
        thresholds = torch.tensor(thresholds, device=device)
        material_volumes = apply_thresholds_split(R0, thresholds, device=device)

        if attenuations is None:
            if E_m == 3: # This is as in the original paper:
                attenuations = torch.stack([torch.tensor([0.2, 1.0, 5.0]) for i in range(N)]).to(device)
            else:
                attenuations = torch.stack([torch.linspace(0.1, 5.0, E_m) for i in range(N)]).to(device)
            for i in range(N):
                mean_atten = R0[material_volumes[i].to(torch.bool)].mean()
                attenuations[i] *= mean_atten

    # Minimum value for parameters:
    eps = 1e-10
    R_i = R0.to(device)
    attenuations.requires_grad = True
    spectrum.requires_grad = True
    for step in trange(steps):
        # Update segmentation thresholds
        if step > 0: # For step = 0, use Otsu initialized thresholds.
            thresholds = bruteforce_thresholds_argmin(thresholds, y_meas, A, R_i, energies, spectrum,
                                                      attenuations, bins=thr_bins, losses=None)#Don't pass losses here

            # Update current segmentations:
            with torch.no_grad():
                material_volumes = apply_thresholds_split(R_i, thresholds, device=device)

        # Update attenuations and spectrum
        args = [material_volumes, y_meas, A, energies, losses]
        torch_minimize([{'params': attenuations, 'lr': lrs[0]},
                    {'params': spectrum, 'lr': lrs[1]}],
                    ISP_atten_objective,
                    iterations=lmin_steps,
                    lr=0.0,
                    func_args=args,
                    constraints=[0.0, torch.inf])

        # Updating reconstruction guess:
        with torch.no_grad():
            y_simulated_poly = spectral_projection_atten(A,
                                                         energies,
                                                         spectrum,
                                                         attenuations,
                                                         material_volumes).squeeze(0)

            # Calculate reference attenuations
            material_projs = [A(material_volumes[i]).squeeze(0) for i in range(material_volumes.shape[0])]
            B = torch.zeros(size=(N,N))
            for i in range(N):
                for j in range(i+1):
                    val = material_projs[i]*material_projs[j]
                    val = val.sum()
                    B[i,j] = val
                    B[j,i] = val
            V = torch.zeros(size=(N,))
            for i in range(N):
                V[i] = -1.0 * (-y_simulated_poly * material_projs[i]).sum()
            atten_ref = torch.matmul(torch.linalg.pinv(B), V)

            for i in range(N):
                material_projs[i] *= atten_ref[i]

            y_simulated_mono = sum(material_projs)

            y_corrected = y_meas + (y_simulated_mono - y_simulated_poly)

            if cone:
                R_i = fdk(A, y_corrected.unsqueeze(0), padded=True)
            else:
                R_i = fbp(A, y_corrected.unsqueeze(0), padded=True)

    return y_corrected, spectrum, attenuations, thresholds


###   Generic end-to-end gradient method >>>>

def GG_objective_func(args, y_meas, A, R_i, energies, losses=None):
    thresholds = args[0]['params'][0]
    attenuations = args[1]['params'][0]
    spectrum =  args[2]['params'][0]
    material_volumes = soft_thresholds_split(R_i, thresholds, gamma=100.0, device=R_i.device)

    y_simulated = spectral_projection_atten(A,
                                            energies,
                                            spectrum,
                                            attenuations,
                                            material_volumes)
    loss = torch.nn.functional.mse_loss(y_simulated, y_meas.unsqueeze(0))
    if losses is not None:#Optionally track losses:
        losses.append(loss.item())
    return loss


def global_gradient_method(y_meas, A, N, energies, spectrum, attenuations=None, cone=False, steps=1, losses=None, local_min_steps=100, lrs=[1e-3, 1e-1, 1e-1]):
    r"""
    y_meas is the input measures sinogram (which has beamhardening artifacts), A is the
    tomosipo projector operator, N is the number of materials, spectrum is the beam spectrum
    NOTE: This assumes y_meas is flat-fielded, and -log(
    """
    # Nr of energy bins
    E_m = spectrum.shape[0]
    device = spectrum.device

    # Initial reconstrunction
    with torch.no_grad():
        if cone:
            R0 = fdk(A, y_meas.unsqueeze(0), padded=True).to(device)
        else:
            R0 = fbp(A, y_meas.unsqueeze(0), padded=True).to(device)

    # Compute initial reasonable thresholds
    with torch.no_grad():
        thresholds = threshold_multiotsu(R0.cpu().detach().numpy(),
                                         classes=N,
                                         nbins=128)
        thresholds = torch.tensor(thresholds, device=device)
        material_volumes = apply_thresholds_split(R0, thresholds, device=device)

        if attenuations is None:
            if E_m == 3: # This is as in the original paper:
                attenuations = torch.stack([torch.tensor([0.2, 1.0, 5.0]) for i in range(N)]).to(device)
            else:
                attenuations = torch.stack([torch.linspace(0.1, 5.0, E_m) for i in range(N)]).to(device)
            for i in range(N):
                mean_atten = R0[material_volumes[i].to(torch.bool)].mean()
                attenuations[i] *= mean_atten

    # Minimum value for parameters:
    eps = 1e-8
    R_i = R0
    thresholds.requires_grad = True
    attenuations.requires_grad = True
    spectrum.requires_grad = True
    for step in trange(steps):
        # Update segmentation thresholds, attenuation, spectrum in one go.
        args = [y_meas, A, R_i, energies, losses]
        torch_minimize([{'params': thresholds, 'lr': lrs[0]},
                        {'params': attenuations, 'lr': lrs[1]},
                        {'params': spectrum, 'lr': lrs[2]}],
                       GG_objective_func,
                       iterations=local_min_steps,
                       lr=0.0,
                       func_args=args,
                       constraints=[eps, torch.inf])

        # Update current segmentations:
        with torch.no_grad():
            material_volumes = apply_thresholds_split(R_i, thresholds, device=device)

        # Updating reconstruction guess:
        with torch.no_grad():
            y_simulated_poly = spectral_projection_atten(A,
                                                         energies,
                                                         spectrum,
                                                         attenuations,
                                                         material_volumes).squeeze(0)

            # Calculate reference attenuations
            material_projs = [A(material_volumes[i]).squeeze(0) for i in range(material_volumes.shape[0])]
            B = torch.zeros(size=(N,N))
            for i in range(N):
                for j in range(i+1):
                    val = material_projs[i]*material_projs[j]
                    val = val.sum()
                    B[i,j] = val
                    B[j,i] = val
            V = torch.zeros(size=(N,))
            for i in range(N):
                V[i] = -1.0 * (-y_simulated_poly * material_projs[i]).sum()
            atten_ref = torch.matmul(torch.linalg.pinv(B), V)

            for i in range(N):
                material_projs[i] *= atten_ref[i]

            y_simulated_mono = sum(material_projs)

            y_corrected = y_meas + y_simulated_mono - y_simulated_poly

            if cone:
                R_i = fdk(A, y_corrected.unsqueeze(0), padded=True)
            else:
                R_i = fbp(A, y_corrected.unsqueeze(0), padded=True)
    return y_corrected, spectrum, attenuations, thresholds
