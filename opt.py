#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[ ]:




