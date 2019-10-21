#!/usr/bin/env python
# coding: utf-8

# In[65]:



from collections import namedtuple
import numpy as np



Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""


def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    
    nfev = 0
    cost = []
    x = x0.copy()
    lmbd = lmbd0
    residual = y - f(*x)
    jac = j(*x)
    
    def delta(x, jac, lmbd):
        grad = jac.T @ (y - f(*x))
        return np.linalg.solve(jac.T @ jac + lmbd * np.identity(np.shape(grad)[0]), grad)
          
    def F(x):
        return 0.5 * (y - f(*x)) @ ( y - f(*x))
    
    while np.linalg.norm(delta(x, jac, lmbd)) > tol:   
        cost.append(F(x))     
        nfev += 1
        gradnorm = np.linalg.norm(jac.T @ (y - f(*x))) 
        F1 = F(x + delta(x, j(*x), lmbd))
        F2 = F(x + delta(x, j(*x), lmbd/nu))
        if F2 <= F(x):
            lmbd/=nu 
        else:
            if F1 <=F (x):
                lmbd = lmbd
            else:
                while F1>=F(x) and np.linalg.norm(delta(x, jac, lmbd))>tol:
                    lmbd*=nu
                    
        x+=delta(x, j(*x), lmbd)
   
    return Result(nfev, cost, gradnorm, x)

def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    nfev = 0 
    cost = [] 
    x=x0.copy()
    delta_x = np.full_like(x0, 2*tol) 
    while np.linalg.norm(delta_x) > tol:
        nfev += 1


# In[77]:


import numpy as np
import scipy as sc
from scipy import integrate
import matplotlib.pyplot as plt
import json


x0 = np.array([0.5, 50])


def load():
    with open('/Users/ananasokeanov/Downloads/jla_mub.txt') as fin:
        lines = fin.readlines()[1:]
            
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n','')
                
        dat = []
        for line in lines:
            words = line.split()
            dat.append(np.array(list(map(float, words))))

        dnu = dat[:, 1].astype('float64')
        dz = dat[:, 0].astype('float64')


def j(x):
    def integrand1(z_):
        return (-1+(1+z_)**3)/(2*(x[0]+(1-x[0])*(1+z_)**3)**(3/2))
    def integrand2(z_):
        return (1/(x[0]+(1-x[0])*(1+z_)**3)**(1/2))
    j_ = np.asarray(dat)

    for i in range(j_.shape[0]):
        j_[i,0] = 5*integrate.quad(integrand1, 0, dz[i])[0]/integrate.quad(integrand2, 0, dz[i])[0]/np.log(10)
        j_[i,1] = -5/x[1]/np.log(10)
    return (j_)


def nu_z(x):
    def integrand(z_, i):
        return 1 / (x[0] + (1 - x[0]) * (1 + z_) ** 3) ** (1 / 2)
    nu_ = np.empty(len(dnu), dtype=float)
    for i in range(len(nu_)):
        nu_[i] = 5*np.log10(integrate.quad(integrand, 0, dz[i], args=i)[0]*3*10**11*(1+dz[i])/x[1])-5
    return nu_


def plot_nu_z():
    
    lm = opt.lm(dnu, nu_z, j, x0)
    x_lm = lm.x
    plt.plot(dz, nu_z(x_lm), label = 'optimized', color = 'blue')
    plt.plot(dz, dnu, '.', label = 'data', color = 'pink')
    plt.title('метод Левенберга—Марквардта')
    plt.xlabel('красное смещение')
    plt.ylabel('расстояние')
    plt.text(0.5, 30, 'H0 = ' + str(round(x_lm[1], 1)) + ', Omega = ' + str(round(x_lm[0], 2)))
    plt.legend()
    plt.show()
    
    
    
    gauss = opt.gauss_newton(dnu, nu_z, j, x0)
    gauss_newton(y, f, j, x0, k=1, tol=1e-4)
    x_gauss = gauss.x
    plt.plot(dz, nu_z(x_gauss), label = 'optimized', color = 'green')
    plt.plot(dz, dnu, '.', label = 'data', color = 'yellow')
    plt.title('метода Гаусса-Ньютона')
    plt.xlabel(' красное смещение')
    plt.ylabel(' модуль расстояния')
    plt.text(0.5, 30, 'H0 = ' + str(round(x_gauss[1], 1)) + ', Omega = ' + str(round(x_gauss[0], 2)))
    plt.legend()
    plt.savefig('mu-z.png')
    plt.show()
    plt.pause(2)
    plt.close()
    return x_gauss, gauss, x_lm, lm



def save(x_gauss, gauss, x_lm, lm):
    with open('parametrs.json', 'w') as file:
        json.dump({"Gauss-Newton": {"H0": x_gauss[1], "Omega": x_gauss[0], "nfev": gauss.nfev},
                   "Levenberg-Marquardt": {"H0": x_lm[1], "Omega": x_lm[0], "nfev": lm.nfev}},
                  file, indent=4, separators=(',', ': '))


def plot_cost(gauss, lm):
    plt.semilogy(np.arange(len(gauss.cost)), gauss.cost, '--', label='gauss')
    plt.semilogy(np.arange(len(lm.cost)), lm.cost, '--', label='Levenberg-Markvardt')
    plt.title('cost')
    plt.xlabel('step')
    plt.ylabel('cost')
    plt.legend()
    plt.savefig('cost.png')
    plt.show()
    plt.pause(2)
    plt.close()


def main():
    load()
    x_gauss, gauss, x_lm, lm = plot_nu_z()
    save(x_gauss, gauss, x_lm, lm)
    plot_cost(gauss, lm)
    
main()


# In[ ]:





# In[ ]:




