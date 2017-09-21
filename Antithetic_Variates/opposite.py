import numpy as np
from matplotlib import use
use('Qt5Agg')
import matplotlib.pyplot as plt

#if __name__=='_main__':
d = 2
nb_sim = 120
rho = np.linspace(0., 1., 11)
result = rho.copy()
result[0] = 1./d
result[-1] = 0.
nb_steps = 10000
L = 10
ind=1
for rh in rho[1:-1]:
    print(rh)
    x = np.zeros((nb_sim, d), dtype=int)
    X = np.zeros((nb_sim, d), dtype=int)
    env = np.random.choice([True, False], size=(nb_sim,) + d*(L,), p=[rh, 1-rh])
    env[(slice(None),) + d * (0,)] = False
    ENV = env.copy()
    ENV[(slice(None),) + d * (0,)] = False
    for j in range(nb_steps):
        for k in range(nb_sim):
            pos = np.random.randint(0, L, size=d)
            axis_jump = np.random.randint(0, d)
            sens = 2*np.random.randint(0,2)-1
            npos = pos.copy()
            npos2 = pos.copy()
            npos[axis_jump] = (npos[axis_jump] + sens)%L
            npos2[axis_jump] = (npos2[axis_jump] - sens)%L
            npos3 = npos.copy()
            npos3[axis_jump] = (npos3[axis_jump] + sens)%L
            pos = (k,) + tuple(pos)
            npos = (k,) + tuple(npos)
            npos2 = (k,) + tuple(npos2)
            npos3 = (k,) + tuple(npos3)
            if sum(pos[1:]) != 0  and sum(npos[1:]) != 0:
                var = env[pos]
                env[pos] = env[npos]
                env[npos] = var
                var = ENV[pos]
                ENV[pos] = ENV[npos]
                ENV[npos] = var
            elif sum(pos[1:]) == 0:
                if not env[npos]:
                    x[k, axis_jump] += sens
                    env[k] = np.roll(env[k], -sens, axis=axis_jump)
                if not ENV[npos2]:
                    X[k, axis_jump] -= sens
                    ENV[k] = np.roll(env[k], sens, axis=axis_jump)
            elif sum(npos[1:]) == 0:
                if not env[pos]:
                    x[k, axis_jump] -= sens
                    env[k] = np.roll(env[k], sens, axis=axis_jump)
                if not ENV[npos3]:
                    X[k, axis_jump] += sens
                    ENV[k] = np.roll(env[k], -sens, axis=axis_jump)


    x2 = L**d * np.ndarray.flatten(x)**2 / nb_steps / 2
    X2 = L**d * np.ndarray.flatten(X)**2 / nb_steps / 2
    expect1 = np.sum(x2)/nb_sim/d
    expect2 = np.sum(X2)/nb_sim/d
    var1 = np.sum((x2 - expect1)**2)/(d*nb_sim-1)
    var2 = np.sum((X2 - expect2)**2)/(d*nb_sim-1)
    cov = np.sum((x2 - expect1)*(X2 - expect2))/(d*nb_sim-1)
    eff = (1 + cov / np.sqrt(var1 * var2))/2
    variance = eff * var1
    result[ind] = 0.5*expect1 + 0.5*expect2
    print('efficacite : ', eff)
    print('variance : ', variance)
    print('resultat : ', d*result[ind])

    ind += 1
        
print(result)
print(variance)
plt.plot(rho, d*result, 'b:o')
plt.plot(rho, 1-rho, 'r')
plt.show()

        
