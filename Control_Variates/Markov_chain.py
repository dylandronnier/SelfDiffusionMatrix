#-*- utf8 -*-

###################################
# Author : Dylan Dronnier
# email : dylan.dronnier@laposte.net
###################################

"""
MONTE CARLO WITH CONTROL VARIATES

----------------------------------------



"""

import multiprocessing as mp
import numpy as np
import argparse


###########################################
############  PARAMETERS  #################
###########################################

parser = argparse.ArgumentParser(
    description='Compute the self-diffusion matrix using control variates')

parser.add_argument("Density",
                    type=float,
                    help="Density of particles at which the self-diffusion coefficient is computed")

parser.add_argument("Simulations",
                    type=int,
                    help="Number of draws. Advice: >100",
                    default=round(3e2))

parser.add_argument("--dimension",
                    metavar="D",
                    type=int,
                    help="Dimension of the grid. Default is 2.",
                    default=2)

parser.add_argument("--size",
                    metavar="L",
                    type=int,
                    help="Size of the grid. Default is 16.",
                    default=16)


parser.add_argument("--processes",
                    metavar="N",
                    type=int,
                    help="Number of processes to run. Default is nproc.",
                    default=mp.cpu_count())

args = parser.parse_args()

d = args.dimension           # Dimension
nb_sim = args.Simulations    # Number of paths in the MC simulation
L = args.size                # Size of the box
alpha = args.Density         # Density of particles
nb_steps = round(5e4)        # Number of steps


def simul():
    """
    Simulate
    """
    x = np.zeros(d, dtype=int)
    X = np.zeros(d, dtype=int)
    env = np.random.choice([True, False], size=d*(L,), p=[alpha, 1-alpha])
    env[d * (0,)] = False
    for j in range(nb_steps):
        pos = np.random.randint(0, L, size=d)
        axis_jump = np.random.randint(0, d)
        sens = 2*np.random.randint(0,2)-1
        npos = pos.copy()
        npos[axis_jump] = (npos[axis_jump] + sens)%L
        pos = tuple(pos)
        npos = tuple(npos)
        if sum(pos) != 0  and sum(npos) != 0:
            var = env[pos]
            env[pos] = env[npos]
            env[npos] = var
        elif sum(pos) == 0:
            if not env[npos]:
                x[axis_jump] += sens
                env = np.roll(env, -sens, axis=axis_jump)
            if np.random.binomial(1, 1-alpha):
                X[axis_jump] += sens
        elif sum(npos) == 0:
            if not env[pos]:
                x[axis_jump] -= sens
                env = np.roll(env, sens, axis=axis_jump)
            if np.random.binomial(1, 1-alpha):
                X[axis_jump] -= sens
    return L**d * x**2 / nb_steps / 2, L**d * X**2 / nb_steps / 2


if __name__=='__main__':
    pool = mp.Pool()
    multiple_results = [pool.apply_async(simul, ()) for i in range(nb_sim)]
    x2 = np.ndarray.flatten(np.array([res.get()[0] for res in multiple_results]))
    X2 = np.ndarray.flatten(np.array([res.get()[1] for res in multiple_results]))
    expect1 = np.sum(x2)/nb_sim/d
    expect2 = np.sum(X2)/nb_sim/d
    var1 = np.sum((x2 - expect1)**2)/(d*nb_sim-1)
    var2 = np.sum((X2 - expect2)**2)/(d*nb_sim-1)
    cov = np.sum((x2 - expect1)*(X2 - expect2))/(d*nb_sim-1)
    c = - cov / var2
    eff = (1 - cov**2 / var1 / var2)
    variance = eff * var1
    result = expect1 + c * (expect2 + (alpha - 1)/d)
    print('efficacite : ', eff)
    print('variance : ', variance)
    print('resultat : ', d*result)

        
