# -*- coding: utf-8 -*-
"""
This is a barebones reimplementation of the original admath, to work with ADFs of numpy.ndarray.

The vast majority of the functions have been deleted, to possibly be reimplemented later.

Of the remaining code, almost all of it is copied directly from the original admath, with only a couple lines changed.

The main changes are as follows:
(1) Removed the @vectorize decorator, as it can lead to accidentally creating ndarrays of ADF objects (whereas we want to use ADF objects containing the ndarrays, for efficiency reasons)
(2) use numpy.function instead of math.function (function=log,exp,expm1, etc), so that ndarrays are handled
(3) Don't check for 0 imaginary part (as this is more annoying for ndarray)

author of edits: Jack Kamm (2015)

-----

ORIGINAL HEADER AND LICENSE:

Mathematical operations that generalize many operations from the standard math
and cmath modules so that they also track first and second derivatives.
The basic philosophy of order of type-operations is this:
A. Is X from the ADF class or subclass? 
   1. Yes - Perform automatic differentiation.
   2. No - Is X an array object?
      a. Yes - Vectorize the operation and repeat at A for each item.
      b. No - Let the math/cmath function deal with X since it's probably a base
         numeric type. Otherwise they will throw the respective exceptions.
Examples:
  from admath import sin
  
  # Manipulation of numbers that track derivatives:
  x = ad.adnumber(3)
  print sin(x)  # prints ad(0.1411200080598672)
  # The admath functions also work on regular Python numeric types:
  print sin(3)  # prints 0.1411200080598672.  This is a normal Python float.
Importing all the functions from this module into the global namespace
is possible.  This is encouraged when using a Python shell as a
calculator.  Example:
  import ad
  from ad.admath import *  # Imports tan(), etc.
  
  x = ad.adnumber(3)
  print tan(x)  # tan() is the ad.admath.tan function
The numbers with derivative tracking handled by this module are objects from
the ad (automatic differentiation) module, from either the ADV or the ADF class.

(c) 2013 by Abraham Lee <tisimst@gmail.com>.
Please send feature requests, bug reports, or feedback to this address.

This software is released under a dual license.  (1) The BSD license.
(2) Any other license, as long as it is obtained from the original
author.

"""
from __future__ import division
import math
import cmath
from ad import __author__, ADF, check_auto_diff, _apply_chain_rule

import numpy as np
numpy_installed = True
                               
__all__ = [
    # math/cmath module equivalent functions
    #'sin', 'asin', 'sinh', 'asinh',
    #'cos', 'acos', 'cosh', 'acosh',
    #'tan', 'atan', 'atan2', 'tanh', 'atanh',
    'e', 'pi', 
    #'isinf', 'isnan', 
    #'phase', 'polar', 'rect',
    'exp', 'expm1',
    #'erf', 'erfc',
    #'factorial', 'gamma', 'lgamma',
    'log', 
    #'ln', 'log10', 'log1p',
    #'sqrt', 'hypot', 'pow',
    #'degrees', 'radians',
    #'ceil', 'floor', 'trunc', 'fabs',
    ## other miscellaneous functions that are conveniently defined
    #'csc', 'acsc', 'csch', 'acsch',
    #'sec', 'asec', 'sech', 'asech',
    #'cot', 'acot', 'coth', 'acoth'
    ]

            
### FUNCTIONS IN THE MATH MODULE ##############################################
#
# Currently, there is no implementation for the following math module methods:
# - copysign
# - factorial <- depends on gamma
# - fmod
# - frexp
# - fsum
# - gamma* <- currently uses high-accuracy finite difference derivatives
# - lgamma <- depends on gamma
# - ldexp
# - modf
#
# we'll see if they(*) get implemented

e = math.e
pi = math.pi


def exp(x):
    """
    Return the exponential value of x
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(check_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = exp(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [exp(x)]
        qc_wrt_args = [exp(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [exp(xi) for xi in x]
#        except TypeError:
#         if x.imag:
#             return cmath.exp(x)
#         else:
#             return math.exp(x.real)
        return np.exp(x) ## EDIT (jackkamm): returns np.exp instead of math.exp


def expm1(x):
    """
    Return e**x - 1. For small floats x, the subtraction in exp(x) - 1 can 
    result in a significant loss of precision; the expm1() function provides 
    a way to compute this quantity to full precision::

        >>> exp(1e-5) - 1  # gives result accurate to 11 places
        1.0000050000069649e-05
        >>> expm1(1e-5)    # result accurate to full precision
        1.0000050000166668e-05

    """
    if isinstance(x,ADF):
        ad_funcs = list(map(check_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = expm1(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [exp(x)]
        qc_wrt_args = [exp(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [expm1(xi) for xi in x]
#        except TypeError:
#        return math.expm1(x) 
        return np.expm1(x) ## EDIT: returns np.expm1
    

## EDIT (jackkamm): unlike in original admath, this does NOT take a base argument    
def log(x):
    """
    With one argument, return the natural logarithm of x (to base e).

    With two arguments, return the logarithm of x to the given base, calculated 
    as ``log(x)/log(base)``.
    """
    if base is None:
        return log(x)
    
    if isinstance(x,ADF):
        
        ad_funcs = list(map(check_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = log(x, base)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1./x]
        qc_wrt_args = [-1./x**2]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [log(xi) for xi in x]
#        except TypeError:
#         if x.imag:
#             return cmath.log(x, base)
#         else:
#             return math.log(x.real, base)
        return np.log(x) ## EDIT (jackkamm): returns np.log instead of math.log
