import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm

import NUR_random

seed = 138
print(f'Seed: {seed}')
generator = NUR_random.XORshift(seed)

first_1000 = generator(1000)

plt.scatter(first_1000[:-1], first_1000[1:])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$x_i$')
plt.ylabel(r'$x_{i+1}$')
plt.savefig('plots/ex1_scatter_sequential.pdf')
plt.close()

plt.scatter(range(1000), first_1000)
plt.xlim(0, 1000)
plt.ylim(0, 1)
plt.xlabel(r'$i$')
plt.ylabel(r'$x_i$')
plt.savefig('plots/ex1_scatter_index.pdf')
plt.close()

bins = np.linspace(0, 1, num=21, endpoint=True)
plt.hist(generator(int(1e6)), bins=bins)
plt.xlabel(r'$x$')
plt.ylabel('Counts')
plt.savefig('plots/ex1_hist_uniform.pdf')
plt.close()

# To go from a standard normal to a general normal:
# Multiply by std, add mean
mean = 3
std = 2.4
normals = NUR_random.Box_Muller(1000, generator) * 2.4 + 3

# Plot the histogram
bins = np.linspace(mean - 5*std, mean+5*std, num=21)
plt.hist(normals, bins=bins, density=True, label='Sampled probability density')

# Plot the theoretical distribution
xs = np.linspace(bins[0], bins[-1], num=1000)
ys = (2*np.pi * std**2)**(-0.5) * np.exp(-1 * (xs-mean)**2 / (2*std**2))
plt.plot(xs, ys, label='Theoretical pdf')

# Plot the 1,2,3,4 sigma edges
for sigma_edge in bins[2:-2:2]:
    plt.axvline(sigma_edge, color='gray', alpha=0.5)

plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$P(x)$')
plt.xlim(bins[0], bins[-1])
plt.savefig('plots/ex1_normal_hist.pdf')
plt.close()

Ns = (10**np.linspace(1, 5, num=41)).astype(int)
p_my_KS = np.empty_like(Ns, dtype=np.float64)
p_sp_KS = np.empty_like(Ns, dtype=np.float64)
data = NUR_random.Box_Muller(100000, generator)
for i, N in enumerate(Ns):
    _, p_my_KS[i] = NUR_random.KS_test(data[:N], NUR_random.normal_cdf)
    _, p_sp_KS[i] = kstest(data[:N], 'norm')

plt.scatter(Ns, p_my_KS, label='My results')
plt.scatter(Ns, p_sp_KS, label='Scipy results')
plt.axhline(0.05, color='gray', alpha=0.5)

plt.xscale('log')
plt.xlabel('Number of Gaussian deviates')
plt.ylabel('p-value')
plt.title('Kolmogorov-Smirnov test')
plt.ylim(0, 1)
plt.legend()
plt.savefig('plots/ex1_KS_test.pdf')
plt.close()

from astropy.stats import kuiper
p_Kuiper = np.empty_like(Ns, dtype=np.float64)
p_Kuiper_astro = np.empty_like(Ns, dtype=np.float64)
for i, N in enumerate(Ns):
    _, p_Kuiper[i] = NUR_random.Kuipers_test(data[:N], NUR_random.normal_cdf)
    _, p_Kuiper_astro[i] = kuiper(data[:N], norm.cdf)

plt.scatter(Ns, p_Kuiper, label='My results')
plt.scatter(Ns, p_Kuiper_astro, label='Astropy results')
plt.axhline(0.05, color='gray', alpha=0.5)

plt.xscale('log')
plt.xlabel('Number of Gaussian deviates')
plt.ylabel('p-value')
plt.title("Kuiper's test")
plt.ylim(0, 1)
plt.legend()
plt.savefig('plots/ex1_Kuiper_test.pdf')
plt.close()

randoms_to_test = np.loadtxt('DataFiles/randomnumbers.txt')
p_vals_test = np.empty((Ns.size, 10))
for i, N in enumerate(Ns):
    for j in range(10):
        _, p_vals_test[i, j] = NUR_random.KS_test_2(data[:N], randoms_to_test[:N, j])
for j in range(10):
    plt.scatter(Ns, p_vals_test[:, j], label=f'Set {j+1}')
plt.axhline(0.05/41, color='gray', alpha=0.5)
plt.axhline(0.05/(41*10), color='gray', alpha=0.5)

plt.xscale('log')
plt.xlabel('Amount of numbers tested')
plt.ylabel('p-value')
plt.title('Two-sided KS test')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.legend(loc='lower left')
plt.savefig('plots/ex1_KS_2sided.pdf')
plt.close()
