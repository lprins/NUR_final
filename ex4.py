import numpy as np
import matplotlib.pyplot as plt
import h5py

from integration import open_Romberg_int, Ridders_deriv
from NUR_random import XORshift, Box_Muller
from fft import fftfreq

# Ignore some divisions by zero which are corrected later
np.seterr(all='ignore')

generator = XORshift(138)

# Normalization of the power spectrum
alpha = 1/8

# Cosmological parameters
Omega_L = 0.7
Omega_M = 0.3
# H_0 in 1/yr
H_0 = 7.16E-11

# Converts from Mpc/yr to km/s
v_convert = 977792221673

def output_3D(z, i, positions, velocities, zpos_particles, zvel_particles):
    zpos_particles[i] = positions[0, 0, :10, 2]
    zvel_particles[i] = velocities[0, 0, :10, 2]

    xcoords = positions[..., 0].flat
    ycoords = positions[..., 1].flat
    zcoords = positions[..., 2].flat

    x_center = (xcoords > 31) & (xcoords < 32)
    y_center = (ycoords > 31) & (ycoords < 32)
    z_center = (zcoords > 31) & (zcoords < 32)

    plt.scatter(xcoords[z_center], ycoords[z_center])
    plt.xlim(0, 64)
    plt.ylim(0, 64)

    plt.xlabel('x (Mpc)')
    plt.ylabel('y (Mpc)')
    plt.title(f'z={z:.2f}')
    plt.savefig(f'movies/3D_zeldovich/xy/{i:03d}.png')
    plt.close()

    plt.scatter(xcoords[y_center], zcoords[y_center])
    plt.xlim(0, 64)
    plt.ylim(0, 64)

    plt.xlabel('x (Mpc)')
    plt.ylabel('z (Mpc)')
    plt.title(f'z={z:.2f}')
    plt.savefig(f'movies/3D_zeldovich/xz/{i:03d}.png')
    plt.close()

    plt.scatter(ycoords[x_center], zcoords[x_center])
    plt.xlim(0, 64)
    plt.ylim(0, 64)

    plt.xlabel('y (Mpc)')
    plt.ylabel('z (Mpc)')
    plt.title(f'z={z:.2f}')
    plt.savefig(f'movies/3D_zeldovich/yz/{i:03d}.png')
    plt.close()

def H_H0(a):
    # Calculates Hubble parameter divided by Hubble constant
    # as a function of the scale factor a
    return (Omega_M * a**-3 + Omega_L)**0.5

def growth_factor_calc(a, return_int=False):
    to_integrate = lambda a: (a * H_H0(a))**-3
    integral_result = open_Romberg_int(to_integrate, 0, a, order=8)
    growth_factor = 2.5 * Omega_M * H_H0(a) * integral_result
    if return_int:
        return growth_factor, integral_result
    else:
        return growth_factor

def analytical_deriv(a, integral_result):
    prefactor = 5 * H_0 * Omega_M / (2 * a**2)
    first_term = 1/H_H0(a)
    second_term = 3 * Omega_M * integral_result / (2 * a)
    return prefactor * (first_term - second_term)

def gen_Zeldovich_S_2D(grid_size):
    c_k = np.empty((grid_size, grid_size, 2), dtype=np.complex128)
    Nq_index = grid_size // 2
    for i in range(grid_size):
        for j in range(grid_size // 2 + 1):
            normals = Box_Muller(2, generator)
            if i > grid_size // 2:
                k = ((2 * np.pi / grid_size)**2 * ((grid_size-i)**2 + j**2))**0.5
            else:
                k = ((2 * np.pi / grid_size)**2 * (i**2 + j**2))**0.5
            c_k[i, j] = alpha * (normals[0] - 1j*normals[1]) / (2 * k**3)
            c_k[-i, -j] = c_k[i, j].conjugate()
    freqs = 2 * np.pi * fftfreq(grid_size)
    c_k[:, :, 0] *= 1j * freqs[:, np.newaxis]
    c_k[:, :, 1] *= 1j * freqs[np.newaxis, :]

    # Correctly set the components that should be real
    c_k[0, 0] = 0
    c_k[0, Nq_index] = c_k[0, Nq_index].real * 2
    c_k[0, Nq_index] = c_k[0, Nq_index].real * 2
    c_k[Nq_index, Nq_index] = c_k[Nq_index, Nq_index].real * 2

    S = grid_size**2 * np.fft.ifft2(c_k, axes=(0, 1)).real
    return S

def gen_Zeldovich_S_3D(grid_size):
    c_k = np.empty((grid_size, grid_size, grid_size, 3), dtype=np.complex128)
    Nq_index = grid_size // 2
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size // 2 + 1):
                normals = Box_Muller(2, generator)
                if i > grid_size // 2:
                    i_mag = grid_size - i
                else:
                    i_mag = i
                if j > grid_size // 2:
                    j_mag = grid_size - j
                else:
                    j_mag = j
                k_vec = ((2 * np.pi / grid_size)**2 * (i_mag**2 + j_mag**2 + k**2))**0.5
                c_k[i, j, k] = alpha * (normals[0] - 1j*normals[1]) / (2 * k_vec**3)
                c_k[-i, -j, -k] = c_k[i, j, k].conjugate()
    freqs = 2 * np.pi * fftfreq(grid_size)
    c_k[:, :, :, 0] *= 1j * freqs[:, np.newaxis, np.newaxis]
    c_k[:, :, :, 1] *= 1j * freqs[np.newaxis, :, np.newaxis]
    c_k[:, :, :, 2] *= 1j * freqs[np.newaxis, np.newaxis, :]

    # Correctly set the components that should be real
    for i in (0, Nq_index):
        for j in (0, Nq_index):
            for k in (0, Nq_index):
                c_k[i, j, k] = c_k[i, j, k].real * 2
    c_k[0, 0, 0] = 0

    S = grid_size**3 * np.fft.ifftn(c_k, axes=(0, 1, 2)).real
    return S

z = 50
a = 1 / (1 + z)
growth_factor, integral_result = growth_factor_calc(a, return_int=True)
print(f'D(z=50) = {growth_factor}')

analytical_result = analytical_deriv(a, integral_result)
dD_da, num_der_err = Ridders_deriv(growth_factor_calc, a, 1e-5)
numerical_result = a * H_0 * H_H0(a) * dD_da

print(f'Analytical result for dD/dt: {analytical_result:.10E} yr-1')
print(f'Numerical  result for dD/dt: {numerical_result:.10E} yr-1')
rel_error = abs(numerical_result - analytical_result) / analytical_result
print(f'Relative difference between analytical and numerical derivative: {rel_error:.2E}')

# Amount of particles in 1 dimension
N_1 = 64
init_pos = np.empty((N_1, N_1, 2))
init_pos[:, :, 0] = np.arange(N_1)[:, np.newaxis]
init_pos[:, :, 1] = np.arange(N_1)[np.newaxis, :]

S_2D = gen_Zeldovich_S_2D(N_1)

scale_factors = np.linspace(0.025, 1, num=90)
ypos_particles = np.empty((scale_factors.size, 10))
yvel_particles = np.empty_like(ypos_particles)

for i, a in enumerate(scale_factors):
    z = 1/a - 1
    D, int_result = growth_factor_calc(a, return_int=True)
    dD_dt = analytical_deriv(a, integral_result)
    positions = np.mod(init_pos + S_2D*D, N_1)
    velocities = -1 * a**2 * dD_dt * S_2D

    ypos_particles[i] = positions[0, :10, 1]
    yvel_particles[i] = velocities[0, :10, 1]

    plt.scatter(positions[:, :, 0].flat, positions[:, :, 1].flat,
                s=1, color='k')

    plt.xlim(0, 64)
    plt.ylim(0, 64)

    plt.xlabel('x (Mpc)')
    plt.ylabel('y (Mpc)')
    plt.title(f'z={z:.2f}')
    plt.savefig(f'movies/2D_zeldovich/{i:03d}.png')
    plt.close()
# Rescale positions from the range 0, 64 to -32, 31
rescaled_pos = np.mod(ypos_particles + N_1//2, N_1) - N_1//2
plt.plot(scale_factors, rescaled_pos)
plt.xlim(0, 1)
plt.xlabel('Scale factor $a$')
plt.ylabel('y-coordinate of particles (Mpc)')
plt.savefig('plots/ex4_2D_ypos.pdf')
plt.close()

plt.plot(scale_factors, v_convert * yvel_particles)
plt.xlim(0, 1)
plt.xlabel('Scale factor $a$')
plt.ylabel('y-velocity of particles (km/s)')
plt.savefig('plots/ex4_2D_yvel.pdf')
plt.close()

S_3D = gen_Zeldovich_S_3D(N_1)

init_pos = np.empty((N_1, N_1, N_1, 3))
init_pos[:, :, :, 0] = np.arange(N_1)[:, np.newaxis, np.newaxis]
init_pos[:, :, :, 1] = np.arange(N_1)[np.newaxis, :, np.newaxis]
init_pos[:, :, :, 2] = np.arange(N_1)[np.newaxis, np.newaxis, :]

# position calculation
z = 50
a = 1 / (1 + z)
D, _ = growth_factor_calc(a, return_int=True)
positions = np.mod(init_pos + S_3D*D, N_1)

# Because we will later use a leapfrog integrator
# velocities have to be calculated half a timestep earlier
a -= 0.00005
_, int_result = growth_factor_calc(a, return_int=True)
dD_dt = analytical_deriv(a, int_result)
velocities = -1 * a**2 * dD_dt * S_3D

zpos_particles = np.empty_like(ypos_particles)
zvel_particles = np.empty_like(zpos_particles)

with h5py.File('init.hdf5', 'w') as f:
    f.create_dataset('Coordinates', data=positions.reshape((-1, 3)))
    f.create_dataset('Velocities', data=velocities.reshape((-1, 3)))

for i, a in enumerate(scale_factors):
    z = 1/a - 1
    D, int_result = growth_factor_calc(a, return_int=True)
    dD_dt = analytical_deriv(a, int_result)
    positions = np.mod(init_pos + S_3D*D, N_1)
    velocities = -1 * a**2 * dD_dt * S_3D

    output_3D(z, i, positions, velocities, zpos_particles, zvel_particles)

rescaled_pos = np.mod(zpos_particles + N_1//2, N_1) - N_1//2
plt.plot(scale_factors, rescaled_pos)
plt.xlim(0, 1)
plt.xlabel('Scale factor $a$')
plt.ylabel('z-coordinate of particles (Mpc)')
plt.savefig('plots/ex4_3D_zpos.pdf')
plt.close()

plt.plot(scale_factors, v_convert * zvel_particles)
plt.xlim(0, 1)
plt.xlabel('Scale factor $a$')
plt.ylabel('z-velocity of particles (km/s)')
plt.savefig('plots/ex4_3D_zvel.pdf')
plt.close()
