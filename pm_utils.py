from math import floor
import numpy as np
np.random.seed(121)
import matplotlib.pyplot as plt

import fft
from numba import njit

@njit
def NGP_mass_assignment_3D(positions, grid_size=16):
    masses = np.zeros((grid_size, grid_size, grid_size))
    for pid in range(positions.shape[0]):
        x, y, z = positions[pid]
        # Calculate the 3 indices
        i = floor(x + 0.5) % grid_size
        j = floor(y + 0.5) % grid_size
        k = floor(z + 0.5) % grid_size
        masses[i, j, k] += 1
    return masses

@njit
def NGP_gradient_assignment(positions, gradient):
    assigned_gradients = np.zeros_like(positions)
    grid_size = gradient.shape[0]
    for pid in range(positions.shape[0]):
        x, y, z = positions[pid]
        i = floor(x + 0.5) % grid_size
        j = floor(y + 0.5) % grid_size
        k = floor(z + 0.5) % grid_size
        assigned_gradients[pid] = gradient[i, j, k]
    return assigned_gradients

@njit
def CIC_mass_assignment_3D(positions, grid_size=16):
    masses = np.zeros((grid_size, grid_size, grid_size))
    for pid in range(positions.shape[0]):
        x, y, z = positions[pid]
        i = floor(x)
        j = floor(y)
        k = floor(z)
        ip1 = (i+1) % grid_size
        jp1 = (j+1) % grid_size
        kp1 = (k+1) % grid_size
        dx = x - i
        dy = y - j
        dz = z - k
        tx = 1 - dx
        ty = 1 - dy
        tz = 1 - dz
        masses[i,   j,   k  ] += tx * ty * tz
        masses[i,   j,   kp1] += tx * ty * dz
        masses[i,   jp1, k  ] += tx * dy * tz
        masses[i,   jp1, kp1] += tx * dy * dz
        masses[ip1, j,   k  ] += dx * ty * tz
        masses[ip1, j,   kp1] += dx * ty * dz
        masses[ip1, jp1, k  ] += dx * dy * tz
        masses[ip1, jp1, kp1] += dx * dy * dz
    return masses

def mesh_to_potential(masses):
    grid_size = masses.shape[0]
    overdensity = (masses - masses.mean()) / masses.mean()
    potential = np.array(overdensity, copy=True, dtype=np.complex128)
    fft.fft3(potential)
    ks_1D = fft.fftfreq(grid_size)
    ks_square = (ks_1D.reshape(grid_size, 1, 1)**2 +
                 ks_1D.reshape(1, grid_size, 1)**2 +
                 ks_1D.reshape(1, 1, grid_size)**2)
    ks_square[0, 0, 0] = 1
    potential /= ks_square
    fft.fft3(potential, sign=1)
    potential = potential / (grid_size**3)
    return potential.real

def potential_to_gradient(potential):
    gradient = np.empty(potential.shape + (3,))
    A = np.pad(potential, ((2, 2), (2, 2), (2, 2)), 'wrap')
    gradient[..., 0] = ((A[:-4] -8*A[1:-3] + 8*A[3:-1] - A[4:]) / 12)[:, 2:-2, 2:-2]
    gradient[..., 1] = ((A[:, :-4] -8*A[:, 1:-3] + 8*A[:, 3:-1] - A[:, 4:]) / 12)[2:-2, :, 2:-2]
    gradient[..., 2] = ((A[:, :, :-4] -8*A[:, :, 1:-3] + 8*A[:, :, 3:-1] - A[:, :, 4:]) / 12)[2:-2, 2:-2, :]
    return gradient

@njit
def assign_gradient(positions, gradient):
    assigned_gradients = np.zeros_like(positions)
    grid_size = gradient.shape[0]
    nparticles = positions.shape[0]
    for pid in range(nparticles):
        x, y, z = positions[pid]
        i = floor(x)
        j = floor(y)
        k = floor(z)
        ip1 = (i+1) % grid_size
        jp1 = (j+1) % grid_size
        kp1 = (k+1) % grid_size
        dx = x - i
        dy = y - j
        dz = z - k
        tx = 1 - dx
        ty = 1 - dy
        tz = 1 - dz
        assigned_gradients[pid] += gradient[i,   j,   k  ] * tx * ty * tz
        assigned_gradients[pid] += gradient[i,   j,   kp1] * tx * ty * dz
        assigned_gradients[pid] += gradient[i,   jp1, k  ] * tx * dy * tz
        assigned_gradients[pid] += gradient[i,   jp1, kp1] * tx * dy * dz
        assigned_gradients[pid] += gradient[ip1, j,   k  ] * dx * ty * tz
        assigned_gradients[pid] += gradient[ip1, j,   kp1] * dx * ty * dz
        assigned_gradients[pid] += gradient[ip1, jp1, k  ] * dx * dy * tz
        assigned_gradients[pid] += gradient[ip1, jp1, kp1] * dx * dy * dz
    return assigned_gradients
