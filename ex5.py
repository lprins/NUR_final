from math import floor
import numpy as np
import matplotlib.pyplot as plt

import fft
from pm_utils import *

np.random.seed(121)
positions = np.random.uniform(low=0, high=16, size=(3, 1024))

masses = NGP_mass_assignment_3D(positions.T)

for z in (4, 9, 11, 14):
    plt.matshow(masses[:, :, z], cmap='Greys')
    plt.colorbar()
    plt.xlabel('x (Mpc)')
    plt.ylabel('y (Mpc)')
    plt.savefig(f'plots/ex5_NGP_z{z}.pdf')
    plt.close()

xpositions = np.linspace(0, 16, num=1000, endpoint=False)
val_0 = np.empty_like(xpositions)
val_4 = np.empty_like(xpositions)

for i, xpos in enumerate(xpositions):
    masses = NGP_mass_assignment_3D(np.array([[xpos, 0, 0]]))
    val_0[i] = masses[0, 0, 0]
    val_4[i] = masses[4, 0, 0]

plt.plot(xpositions, val_0, label='Mass in cell 0')
plt.plot(xpositions, val_4, label='Mass in cell 4')
plt.xlabel('x position of particle (Mpc)')
plt.ylabel('Mass')
plt.xlim(0, 16)
plt.legend()
plt.title('Nearest Grid Point assignment')
plt.savefig('plots/ex5_NGP_xpos.pdf')
plt.close()

masses = CIC_mass_assignment_3D(positions.T)

for z in (4, 9, 11, 14):
    plt.matshow(masses[:, :, z], cmap='Greys')
    plt.colorbar()
    plt.xlabel('x (Mpc)')
    plt.ylabel('y (Mpc)')
    plt.savefig(f'plots/ex5_CIC_z{z}.pdf')
    plt.close()

for i, xpos in enumerate(xpositions):
    masses = CIC_mass_assignment_3D(np.array([[xpos, 0, 0]]))
    val_0[i] = masses[0, 0, 0]
    val_4[i] = masses[4, 0, 0]

plt.plot(xpositions, val_0, label='Mass in cell 0')
plt.plot(xpositions, val_4, label='Mass in cell 4')
plt.xlabel('x position of particle (Mpc)')
plt.ylabel('Mass')
plt.xlim(0, 16)
plt.legend()
plt.title('Cloud In Cell assignment')
plt.savefig('plots/ex5_CIC_xpos.pdf')
plt.close()

fs = np.ones(32)
fs[1::2] = -1

fs_myfft = np.array(fs, copy=True, dtype=np.complex128)
fft.fft(fs_myfft)
fs_npfft = np.fft.fft(fs)
if np.allclose(fs_myfft, fs_npfft):
    print('1D FFT is equal to FFTPACK up to machine precision')
else:
    print('Error in 1D FFT')

plt.plot(fs_myfft.real, label='Real part of FT')
plt.plot(fs_myfft.imag, label='Imaginary part of FT')
plt.plot(fs, label='Function values')
plt.xlim(0, 31)
plt.legend()
plt.savefig('plots/ex5_FFT1.pdf')
plt.close()

fs = np.ones((32, 32))
fs[1::2, ::2] = -1
fs[::2, 1::2] = -1

fs_myfft = np.array(fs, copy=True, dtype=np.complex128)
fft.fft2(fs_myfft)
fs_npfft = np.fft.fft2(fs)

if np.allclose(fs_myfft, fs_npfft):
    print('2D FFT is equal to FFTPACK up to machine precision')
else:
    print('Error in 2D FFT')

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.matshow(fs, cmap='Greys')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Function values')

ax2.matshow(fs_myfft.real, cmap='Greys')
ax2.set_xlabel('k_x')
ax2.set_ylabel('k_y')
ax2.set_title('Fourier Transform')
plt.tight_layout()
plt.savefig('plots/ex5_FFT2.pdf')
plt.close()

xs = np.arange(-16, 16)
ys = np.arange(-16, 16)
zs = np.arange(-16, 16)
rs_2 = (xs**2)[:, None, None] + (ys**2)[None, :, None] + (zs**2)[None, None, :]

fs = np.exp(-1 * rs_2/8)

fs_myfft = np.array(fs, copy=True, dtype=np.complex128)
fft.fft3(fs_myfft)
fs_npfft = np.fft.fftn(fs)

if np.allclose(fs_myfft, fs_npfft):
    print('3D FFT is equal to FFTPACK up to machine precision')
else:
    print('Error in 3D FFT')
# Shift FFT so DC term is at center
fs_myfft = np.roll(fs_myfft, 16, axis=(0, 1, 2))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.matshow(np.abs(fs_myfft[:, :, 16]))
ax1.set_title('x-y slice')
ax2.matshow(np.abs(fs_myfft[:, 16, :]))
ax2.set_title('x-z slice')
ax3.matshow(np.abs(fs_myfft[16, :, :]))
ax3.set_title('y-z slice')
plt.savefig('plots/ex5_FFT3.pdf')
plt.close()

masses = CIC_mass_assignment_3D(positions.T)
potential = mesh_to_potential(masses)

for z in (4, 9, 11, 14):
    plt.matshow(potential[:, :, z])
    plt.xlabel('x (Mpc)')
    plt.ylabel('y (Mpc)')
    plt.savefig(f'plots/ex5_pot_z{z}.pdf')
    plt.close()

plt.matshow(potential[:, 8, :])
plt.xlabel('x (Mpc')
plt.ylabel('z (Mpc')
plt.savefig('plots/ex5_pot_xz.pdf')

plt.matshow(potential[8, :, :])
plt.xlabel('y (Mpc')
plt.ylabel('z (Mpc')
plt.savefig('plots/ex5_pot_yz.pdf')

gradient = potential_to_gradient(potential)
assigned_gradients = assign_gradient(positions.T, gradient)

print('Gradients at first 10 particles:')
print(assigned_gradients[:10])
