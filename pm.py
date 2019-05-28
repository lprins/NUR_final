import sys
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
import h5py

from pm_utils import CIC_mass_assignment_3D, mesh_to_potential, potential_to_gradient, assign_gradient

grid_size = 64
Omega_L = 0.7
Omega_m = 0.3
# in yr-1
H_0 = 7.16E-11
a_start = 1/51
delta_a = 0.0001


outfolder = 'sim_snapshots'

N_steps = ceil((1 - a_start) / delta_a)

def load_ICs(filename):
    with h5py.File(filename, 'r') as infile:
        positions = np.array(infile['Coordinates'])
        velocities = np.array(infile['Velocities']) * f(a_start) / H_0
    return positions, velocities

def f(a):
    return (a / (Omega_m + Omega_L * a**3))**0.5

def save_snapshot(a, positions, velocities, filename=None):
    if filename is None:
        filename = f'pm_{a:.4f}.hdf5'
    with h5py.File(f'{outfolder}/{filename}', 'w') as outfile:
        outfile.create_dataset('Coordinates', data=positions)
        outfile.create_dataset('Velocities', data=velocities)

positions, velocities = load_ICs(sys.argv[1])
a = a_start

save_every_n = N_steps // 300

save_snapshot(a, positions, velocities, filename='pm_init.hdf5')
for i in range(N_steps):
    print(f'z = {1/a - 1: 7.3f}, i = {i: 4d}, progress = {i/N_steps: 5.3f}', end='\r')
    masses = CIC_mass_assignment_3D(positions, grid_size)
    # Check units of potential
    potential = (3 * Omega_m / (2 * a)) * mesh_to_potential(masses)
    gradient = potential_to_gradient(potential)
    assigned_gradients = assign_gradient(positions, gradient)

    velocities -= assigned_gradients * f(a) * delta_a
    a_n_half = a + 0.5 * delta_a
    positions += velocities * f(a_n_half) * delta_a / a_n_half**2
    np.mod(positions, grid_size, out=positions)
    if i % save_every_n == 0:
        save_snapshot(a, positions, velocities)
    a += delta_a
save_snapshot(a, positions, velocities, filename='pm_final.hdf5')
