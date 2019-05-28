import numpy as np
from numba import njit

@njit
def bit_reverse(data, lg2_N):
    # Bit reverses elements of data, which should be of size 2**lg2_N
    index = np.zeros(1, dtype=np.int64)
    for i in range(lg2_N):
        # A beautiful result from wikipedia:
        # We can double the k-1 bits permutation, concatenate
        # with 1 added to it and get the k bits permutation
        # https://en.wikipedia.org/wiki/Bit-reversal_permutation
        index = np.concatenate((index*2, index*2 +1))
    # now rearrange the data array
    for i, j in enumerate(index):
        if i > j:
            temp = data[i]
            data[i] = data[j]
            data[j] = temp

@njit
def fft(data, sign=-1):
    '''
    In-place DFT calculation, len(data) should be a power of 2
    Note that NR and slides use sign=1,
    FFTW and FFTPACK (and thus numpy and scipy) use sign=-1
    Flipping sign gives the inverse transform up to a 1/n normalization
    Based on Introduction to Algorithms by CLRS 3rd ed. 30.3
    '''
    N = data.size
    # If N is a power of 2, it has form 1000...,
    # N - 1 has the form 0111...
    # Thus N & (N - 1) == 0 iff N a power of 2
    if N & (N - 1) != 0:
        raise ValueError('Data size must be a power of 2')
    # Note that powers of 2 are exact for floats, so no need to round
    lg2_N = int(np.log2(N))
    bit_reverse(data, lg2_N)
    for s in range(1, lg2_N + 1):
        m = 2**s
        # Calculate principal mth root of unity
        omega_m = np.exp(sign * 2j * np.pi / m)
        for k in range(0, N, m):
            # omega is multiplied by omega_m cycling through
            # the required factors which are all mth roots of unity
            omega = 1
            for j in range(m//2):
                # Apply the Danielson-Lanczos update
                t = omega * data[k + j + m//2]
                u = data[k + j]
                data[k + j] = u + t
                data[k + j + m//2] = u - t
                # Calculate next factor omega
                omega = omega * omega_m

@njit
def fft3(data, sign=-1):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            fft(data[i, j, :], sign)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            fft(data[:, i, j], sign)
    for i in range(data.shape[2]):
        for j in range(data.shape[0]):
            fft(data[j, :, i], sign)
@njit
def fft2(data, sign=-1):
    for i in range(data.shape[0]):
        fft(data[i, :], sign)
    for j in range(data.shape[1]):
        fft(data[:, j], sign)

def fftfreq(N):
    # returns frequencies of FFT transform
    freqs = np.empty(N)
    pos_edge = (N - 1) // 2 + 1
    freqs[:pos_edge] = np.arange(0, pos_edge)
    freqs[pos_edge:] = np.arange(-(N//2), 0)
    return freqs/N
