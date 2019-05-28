import numpy as np

# Tsitouras 5th (4th) order RK method, from 
# https://doi.org/10.1016/j.camwa.2011.06.002
# Coefficients taken from the DifferentialEquations.jl package
# Load A coefficients from file
Tsit5A = np.loadtxt('Tsit5A.txt', dtype=np.float64)
# Hardcode c coefficients
Tsit5c = np.array([0.0, 0.161, 0.327, 0.9, 0.980026, 1.0, 1.0], dtype=np.float64)
# coefficients for 5th and 4th order method
Tsit5b_5 = Tsit5A[-1, :]
Tsit5b_4 = np.array([0.0946808, 0.00918357, 0.487771, 1.2343, -2.70771, 1.86663, 0.0151515],
                    dtype=np.float64)
# No need to calculate 4th order directly, can precalculate terms
Tsit5b_error = Tsit5b_5 - Tsit5b_4
def Tsit5_step(x, h, y, f_dy, k0):
    ks = np.zeros((7, k0.size))
    xs = x + h * Tsit5c
    ks[0] = k0
    for i in range(1, 7):
        y_val = y + h * Tsit5A[i] @ ks
        ks[i] = f_dy(xs[i], y_val)
    new_y = y + h * Tsit5b_5 @ ks
    err = h * Tsit5b_error @ ks
    return new_y, err, ks[-1]

def Tsit5_adaptive(x1, x2, y0, f_dy, atol=1e-6, rtol=1e-6, maxsteps=10_000):
    S = 0.95
    step_order = 5
    xs = np.empty(maxsteps+1)
    ys = np.empty((maxsteps+1, y0.size))
    xs[0] = x1
    ys[0] = y0
    k0 = f_dy(xs[0], ys[0])
    h = (x2 - x1) / maxsteps
    i = 0
    for nstep in range(maxsteps):
        new_y, err, k_new = Tsit5_step(xs[i], h, ys[i], f_dy, k0)
        scale = atol + rtol * np.maximum(np.abs(new_y), np.abs(ys[i]))
        norm_err = (np.sum((err/scale)**2) / err.size)**0.5
        if norm_err <= 1:
            xs[i+1] = xs[i] + h
            ys[i+1] = new_y
            i += 1
            k0 = k_new
            if xs[i] >= x2:
                break
            h *= min(5, S * norm_err**(-1/step_order))
            if xs[i] + h > x2:
                h = x2 - xs[i]
        else:
            h *= max(.2, S * norm_err**(-1/step_order))
    if xs[i] < x2:
        print('max steps reached before reaching x2')
    return xs[:i+1], ys[:i+1]
