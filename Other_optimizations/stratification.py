#-*- utf8 -*-

#####################################
# Author : Dylan Dronnier
# email : dylan.dronnier@laposte.net
#####################################

"""
MONTE CARLO, SIMULTANEOUS COMPUTATION

----------------------------------------

We compute the value of the self-diffusion coefficient for all density
\alpha in [0,1] using a greedy stratified sampling approach.

"""

import multiprocessing as mp
import numpy as np
import argparse


###########################################
############  PARAMETERS  #################
###########################################

parser = argparse.ArgumentParser(
    description='Compute the self-diffusion matrix using control variates')

parser.add_argument("Number of points",
                    type=int,
                    help="Number of points "
                    default=

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



def simul(N):
    """
    Simulate

    -----------------

    N : number of particules to simulate


    """
    env = -np.ones(d*(L,), dtype=int)
    for i in range(N):
        pos = (0,)*d
        while env[pos] >= 0:
            pos = np.random.randint(
        env[pos] = i
    x = np.zeros((cpt, d), dtype=int)
    for j in range(nb_steps):
        select = np.random.randint(0, cpt) 
        axis_jump = np.random.randint(0, d)
        sens = 2*np.random.randint(0,2)-1
        pos = env[tuple(x[select]%L)]
        npos = pos.copy()
        npos[axis_jump] = (npos[axis_jump] + sens)%L
        pos = tuple(pos)
        npos = tuple(npos)
        if env[npos]==-1:
            var = env[pos]
            env[pos] = env[npos]
            env[npos] = var
            x[select, axis_jump] += sens
    return L**d * x**2 / nb_steps / 2,

if __name__=='__main__':
    pool = mp.Pool()
    multiple_results = [pool.apply_async(simul, ()) for i in range(nb_sim)]
    x2 = np.ndarray.flatten(np.array([res.get()[0] for res in multiple_results]))
    expect = np.sum(x2)/nb_sim/d
    var = np.sum((x2 - expect)**2)/(d*nb_sim-1)
    print('variance : ', var)
    print('resultat : ', d*expect)
