import numpy as np
import matplotlib.pyplot as plt

from ODE import Tsit5_adaptive

def analytic_solution(ts, D, D_prime):
    # Analytic solution of the EdS linear growth equation
    # Initial conditions are required at t=1
    A = (D + D_prime) * 3 / 5
    B = D - A
    return A*ts**(2/3) + B*ts**-1

def calc_plot_case(D, D_prime, label):
    def f_dD(t, pars):
        # Set up the differential equation
        dD_dt = pars[1]
        d2D_dt2 = 2*pars[0]/(3*t**2) - 4*pars[1]/(3*t)
        return np.array((dD_dt, d2D_dt2))
    t0 = 1
    t1 = 1000
    init = np.array((D, D_prime))
    ts, ys_numeric = Tsit5_adaptive(t0, t1, init, f_dD,
                                    maxsteps=200, atol=1e-7, rtol=1e-5)
    Ds_numeric = ys_numeric[:, 0]
    Ds_analytic = analytic_solution(ts, D, D_prime)

    plt.loglog(ts, Ds_analytic, label='Analytic solution', color='k')
    plt.plot(ts, Ds_numeric, label='Numeric solution', color='b')
    plt.scatter(ts, Ds_numeric, color='b')

    plt.xlabel('t [yr]')
    plt.ylabel('Growth factor D')
    plt.title(f"D={D} D'={D_prime}")
    plt.legend()

    plt.savefig(f'plots/ex3_Growth_ODE_case_{label}.pdf')
    plt.close()

calc_plot_case(3, 2, '1')
calc_plot_case(10, -10, '2')
calc_plot_case(5, 0, '3')
