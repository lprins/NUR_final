import numpy as np
from sorting import mergesort

# Chebyshev coefficients for approximating the complementary error function
# Taken from NR 6.2.2
_erfc_Cheb_coeff = np.array([-1.3026537197817094, 6.4196979235649026e-1,
                             1.9476473204185836e-2, -9.561514786808631e-3,
                             -9.46595344482036e-4, 3.66839497852761e-4,
                             4.2523324806907e-5, -2.0278578112534e-5,
                             -1.624290004647e-6, 1.303655835580e-6,
                             1.5626441722e-8, -8.5238095915e-8,
                             6.529054439e-9, 5.059343495e-9,
                             -9.91364156e-10, -2.27365122e-10,
                             9.6467911e-11, 2.394038e-12,
                             -6.886027e-12, 8.94487e-13,
                             3.13092e-13, -1.12708e-13,
                             3.81e-16, 7.106e-15,
                             -1.523e-15, -9.4e-17,
                             1.21e-16, -2.8e-17])
def _erfc_Cheb(x):
    # Calculates erfc(x) for x>=0 using erfc(x) = t exp(-x**2 + P(t))
    # with t = 2 / (2+z) and P a polynomial defined by
    # the coefficients _erfc_Cheb
    ncoeffs = len(_erfc_Cheb_coeff)
    t = 2 / (2 + x)
    # Change of variables to -1 <= ty <= 1
    # Also multiply by 2 for using Clenshaws recurrence
    ty = 4*t - 2
    d = 0
    dd = 0
    for i in range(ncoeffs-1, 0, -1):
        temp = d
        d = ty * d - dd + _erfc_Cheb_coeff[i]
        dd = temp
    poly_total = 0.5 * (_erfc_Cheb_coeff[0] + ty * d) - dd
    return t * np.exp(-1*x**2 + poly_total)

def erfc(x):
    # Calculate the complementary error function
    if x < 0:
        return 2 - _erfc_Cheb(-x)
    else:
        return _erfc_Cheb(x)

def normal_cdf(x):
    # Calculates the cdf for the standard normal distribution
    return 0.5 * erfc(-1 * 2**-0.5 * x)

def KS_cdf(z):
    # Calculates Kolmogorov-Smirnov cdf using approximation 
    # from the lecture slides
    if z == 0:
        return 0
    elif z < 1.18:
        exp_term = np.exp(-1 * np.pi**2 / (8 * z**2))
        return (2*np.pi)**0.5 * z**-1 * (exp_term + exp_term**9 + exp_term**25)
    else:
        exp_term = np.exp(-2 * z**2)
        return 1 - 2 * (exp_term - exp_term**4 + exp_term**9)

def KS_test(data, cdf):
    data = np.copy(data)
    mergesort(data)
    N = len(data)
    D = 0
    prev_cdf = 0
    for i, sample in enumerate(data):
        data_cdf = (i + 1) / N
        dist_cdf = cdf(sample)
        distance = max(np.abs(data_cdf - dist_cdf), np.abs(prev_cdf - dist_cdf))
        if distance > D:
            D = distance
        prev_cdf = data_cdf
    p_val = 1 - KS_cdf(D * (N**0.5 + 0.12 + 0.11 * N**-0.5))
    return D, p_val

def KS_test_2(data1, data2):
    data1 = np.copy(data1)
    data2 = np.copy(data2)
    mergesort(data1)
    mergesort(data2)
    N1 = len(data1)
    N2 = len(data2)

    D = 0

    i1 = i2 = 0
    f1 = f2 = 0
    while i1 < N1 and i2 < N2:
        d1 = data1[i1]
        d2 = data2[i2]
        if d1 <= d2:
            i1 += 1
            f1 = i1 / N1 
        if d2 <= d1:
            i2 += 1
            f2 = i2 / N2
        distance = np.abs(f2 - f1)
        if distance > D:
            D = distance
    N_eff_sqrt = np.sqrt((N1 * N2) / (N1 + N2))
    p_val = 1 - KS_cdf(D * (N_eff_sqrt + 0.12 + 0.11 / N_eff_sqrt))
    return D, p_val

def Kuipers_cdf(z):
    # Upper tail approximation for the Kuipers test value
    # Based on Stephens (1970)
    if z < 1:
        return 1
    return (8*z**2 - 2) * np.exp(-2*z**2)

def Kuipers_test(data, cdf):
    # Note that the p values returned are only accurate if they are small
    # Accurate within 2 decimal places for p < 0.74
    # Thus, a rejection of H0 is real, but the p-values are not distributed
    # as could be expected
    data = np.copy(data)
    mergesort(data)
    N = len(data)
    D_plus = 0
    D_minus = 0
    prev_cdf = 0
    for i, sample in enumerate(data):
        data_cdf = (i + 1) / N
        dist_cdf = cdf(sample)
        distance_plus = data_cdf - dist_cdf
        distance_minus = dist_cdf - prev_cdf
        if distance_plus > D_plus:
            D_plus = distance_plus
        if distance_minus > D_minus:
            D_minus = distance_minus
        prev_cdf = data_cdf
    D = D_minus + D_plus
    p_val = Kuipers_cdf(D * (N**0.5 + 0.155 + 0.24 * N**-0.5))
    return D, p_val

def Pearson_corr(xs):
    x = xs[:-1]
    y = xs[1:]
    return ((x*y).mean() - x.mean() * y.mean()) / (x.std() * y.std())

def rejection_generator(pdf, x_min, x_max, p_max, generator):
    rand_x = lambda: (x_max - x_min) * generator() + x_min
    rand_y = lambda: p_max * generator()
    while True:
        x = rand_x()
        y = rand_y()
        if y <= pdf(x):
            return x

def unit_sphere(N, generator):
    thetas = np.arccos(1 - 2 * generator(N))
    phis = 2 * np.pi * generator(N)
    return thetas, phis

class XORshift:
    def __init__(self, seed):
        np.seterr(over='ignore')
        self._state1 = np.uint64(seed)
        self._state2 = np.uint64(seed)
        self._a = np.uint32(4294957665)
        self._high32 = np.uint32(0xffffffff)
        # Some shift constants, initialized here to avoid type casting
        # for every invocation of the RNG
        self._shift1 = np.uint64(17)
        self._shift2 = np.uint64(31)
        self._shift3 = np.uint64(8)
        self._4byte = np.uint64(32)
        if seed == 0:
            raise ValueError('Seed should not be 0!')

    def _generate(self):
        x = self._state1
        y = self._state2
        # We need to convert the shift amounts to uint64 because of a numpy bug
        # See https://github.com/numpy/numpy/issues/2524
        x ^= x >> self._shift1
        x ^= x << self._shift2
        x ^= x >> self._shift3
        y = self._a * (y & self._high32) + (y >> self._4byte)
        self._state1 = x
        self._state2 = y
        return x ^ y

    def __call__(self, n=None, uniform=True):
        if n is None:
            val = self._generate()
        else:
            val = [self._generate() for _ in range(n)]
            val = np.array(val)
        if uniform:
            val = val / (2**64 - 1)
        return val

def Box_Muller(N, generator):
    # Generates N standard normal deviates
    # N & 1 is equivalent to but faster than N % 2
    # Box Muller method generates even number of deviates
    if N & 1 == 0:
        randoms = generator(N)
    else:
        randoms = generator(N + 1)
    normals = np.empty_like(randoms)
    even_randoms = randoms[::2]
    odd_randoms = randoms[1::2]

    rs = (-2 * np.log(even_randoms))**0.5
    normals[::2] = rs * np.cos(2 * np.pi * odd_randoms)
    normals[1::2] = rs * np.sin(2 * np.pi * odd_randoms)
    # Make sure we return an odd number of deviates if needed
    return normals[:N]
