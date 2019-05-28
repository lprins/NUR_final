import numpy as np
import matplotlib.pyplot as plt

from NUR_random import XORshift, Box_Muller

generator = XORshift(234)
print(f'Seed: 234')

def gen_random_field_2D(N, n):
    field = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N//2 + 1):
            k = (i**2 + j**2)**0.5
            try:
                std = k**(n / 2)
                normals = Box_Muller(2, generator) * std
                amp = normals[0] + 1j * normals[1]
                field[i, j] = amp
                field[-i, -j] = amp.conjugate()
            except ZeroDivisionError:
                field[i, j] = 0
    field[0, 0] = 0
    field[0, N//2] = 2 * field[0, N//2].real
    field[N//2, N//2] = 2 * field[N//2, N//2].real
    field[N//2, 0] = 2 * field[N//2, 0].real
    field = np.fft.ifft2(field)
    return field.real

for n in [-1, -2, -3]:
    field = gen_random_field_2D(1024, n)
    plt.matshow(field, origin='lower')
    plt.xlabel('x (Mpc)')
    plt.ylabel('y (Mpc)')
    plt.title(f'n = {n}')
    plt.savefig(f'plots/ex2_n_{abs(n)}.pdf')
