#!/usr/bin/env python3

from collections import namedtuple


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Results of optimization

Attributes
----------
nfev : int
    Function call count
cost : 1-d array
    Values of the cost function 0.5 sum(y - f)^2 on every iteration step.
    Length of this array is less than nfev in the case of LM method
gradnorm : float
    Norm of gradient for the last iteration step
x : 1-d array
    Final vector, minimising cost function
"""
