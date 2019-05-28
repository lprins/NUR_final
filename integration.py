import numpy as np
from scipy.special import roots_legendre, roots_chebyt

def trapezoid_rule(xs, fs, dx):
    # Trapezoid rule: integral from x0 to xN-1 is
    # (0.5f0 + f1 + f2 + ... + 0.5fN-1)dx equivalent to
    # dx(sum(fs) - 0.5(f0 + fN-1))
    return dx * (np.sum(fs) - 0.5 * (fs[0] + fs[-1]))

def Simpson_rule(xs, fs, dx):
    odd_sum = np.sum(fs[1:-1:2])
    even_sum = np.sum(fs[2:-1:2])
    return (fs[0] + fs[-1] + 4 * odd_sum + 2 * even_sum) * dx / 3

def Romberg_int(f, a, b, order=6):
    def combine(MA, LA, m):
        # Combines approximations, MA most accurate, LA least accurate
        prefactor = 1/(4**m - 1)
        return MA + prefactor * (MA - LA)
    n_points = 1 + 2**order
    xs = np.linspace(a, b, num=n_points, endpoint=True)
    fs = f(xs)
    approximations = np.tile(np.nan, (order + 1, order + 1))
    # Initial approximations of integral
    for n in range(order + 1):
        stride = 2**(order - n)
        dx = (b-a) / 2**n
        approximations[n, 0] = trapezoid_rule(xs[::stride], fs[::stride], dx)
    # Combine the approximations
    for m in range(1, order + 1):
        for n in range(m, order + 1):
            MA = approximations[n, m-1]
            LA = approximations[n-1, m-1]
            approximations[n, m] = combine(MA, LA, m)
    return approximations[-1, -1]

def open_Romberg_int(f, a, b, order=4):
    def combine(MA, LA, m):
        prefactor = 1/(9**m - 1)
        return MA + prefactor * (MA - LA)
    n_points = 3**order
    xs = np.linspace(a, b, num=n_points*2+1, endpoint=True)[1::2]
    fs = f(xs)
    approximations = np.tile(np.nan, (order + 1, order + 1))
    # Initial approximations of integral
    for n in range(order + 1):
        stride = 3**(order - n)
        start = stride//2
        dx = (b-a) / 3**n
        approximations[n, 0] = np.sum(fs[start::stride]) * dx
    # Combine the approximations
    for m in range(1, order + 1):
        for n in range(m, order + 1):
            MA = approximations[n, m-1]
            LA = approximations[n-1, m-1]
            approximations[n, m] = combine(MA, LA, m)
    return approximations[-1, -1]

def GLegendre_quad(f, n):
    # Integrates the function f from -1 to 1,
    # using n-point Gauss-Legendre quadrature
    xs, weights = roots_legendre(n)
    fs = f(xs)
    return np.sum(fs * weights)

def GCheby_quad(f, n):
    # Integrates the function f from -1 to 1,
    # with weight function (1 - x**2)**(-1/2)
    xs, weights = roots_chebyt(n)
    fs = f(xs)
    return np.sum(fs * weights)

def central_diff_deriv(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def Ridders_deriv(f, x, h, target_err=1e-10, max_order=5):
    def CDD(step_size):
        return central_diff_deriv(f, x, step_size)
    # factor by which to reduce step size
    d = 1.4
    err = np.inf
    ans = np.nan
    D = np.tile(np.nan, (max_order, max_order))
    D[0][0] = CDD(h)
    for i in range(1, max_order):
        h /= d
        D[i][0] = CDD(h)
        for j in range(1, i+1):
            d_factor = d**(2 * j)
            D[i][j] = (d_factor * D[i][j-1] - D[i-1][j-1]) / (d_factor  - 1)
            new_err = max(abs(D[i][j] - D[i][j-1]), abs(D[i][j] - D[i-1][j-1]))
            if new_err < err:
                err = new_err
                ans = D[i][j]
                if err < target_err:
                    return ans, err
        if abs(D[i][i] - D[i-1][i-1]) > 2 * err:
            return ans, err
    return ans, err
