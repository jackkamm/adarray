from .ad import *
from .ad import ADF, _get_variables, to_auto_diff, _apply_chain_rule, _is_constant, constant, null, get_order, set_order
from .ad import admath

import numpy as np
from numbers import Number
import scipy, scipy.signal, scipy.sparse
from scipy.sparse.linalg import expm_multiply

def _make_derivs_dicts():
    if get_order() == 1:
        # return dictionary for first order terms
        return {}, None, None
    elif get_order() == 2:
        # dictionaries for first order, second order, cross terms
        return {}, {}, {}

def array(x):
    if isinstance(x, ADF):
        return x

    if isinstance(x, Number):
        return constant(x)

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if np.issubdtype(x.dtype, np.number):
        return constant(x)

    # get the variables to differentiate against
    adentries = []
    for i,xi in np.ndenumerate(x):
        if isinstance(xi, ADF):
            adentries.append((i,xi))
        elif not isinstance(xi,Number):
            raise TypeError(str((i,xi)))
    variables = _get_variables([xi for _,xi in adentries])

    # initialize the dictionaries of derivatives
    d_dicts = lc,qc,cp = _make_derivs_dicts()
    d_dicts = [d for d in d_dicts if d is not None]

    if variables:
        # fill the dictionaries of derivatives
        for i,xi in adentries:
            for xi_d, x_d in zip((xi._lc, xi._qc, xi._cp)[:len(d_dicts)], d_dicts):
                for k in xi_d:
                    if k not in x_d:
                        x_d[k] = np.zeros(x.shape)
                    x_d[k][i] = xi_d[k]

    x_old = x
    x = np.zeros(x.shape)
    for i,xi in np.ndenumerate(x_old):
        if isinstance(xi,ADF):
            x[i] = xi.x
        elif isinstance(xi, Number):
            x[i] = xi
        else:
            raise Exception

    return ADF(x, lc, qc, cp)


'''add array functionality to ADF'''

'''add shape and length to ADF'''
@property
def ad_shape(self):
    return self.x.shape

def ad_len(self):
    return self.x.__len__()

ADF.shape = ad_shape
ADF.__len__ = ad_len

''' apply f to x and all its derivatives '''
def ad_apply(self, f, *args, **kwargs):
    ret_x = f(self.x, *args, **kwargs)
    if self._lc is null:
        return constant(ret_x)

    d_dicts = lc, qc, cp = _make_derivs_dicts()
    d_dicts = [d for d in d_dicts if d is not None]
    for ret_d, x_d in zip(d_dicts, (self._lc, self._qc, self._cp)[:len(d_dicts)]):
        for v in x_d:
            ret_d[v] = f(x_d[v], *args, **kwargs)
    return ADF(ret_x, lc, qc, cp)

ADF.apply = ad_apply

''' __getitem__ and __setitem__'''
def ad_getitem(self, *args, **kwargs):
    return self.apply(np.ndarray.__getitem__, *args, **kwargs)

def ad_setitem(self, key, value):
    if not _is_constant(key):
        ## TODO: implement non-constant case!
        raise NotImplementedError
    self.x[key] = value
    for derivatives in (self._lc, self._qc, self._cp):
        if derivatives is not None:
            for direction in derivatives:
                derivatives[direction][key] = 0.0

ADF.__getitem__ = ad_getitem
ADF.__setitem__ = ad_setitem


''' sum function and method '''
def sum(x, *args, **kwargs):
    if isinstance(x, ADF):
        return x.apply(np.sum, *args, **kwargs)
    return np.sum(x, *args, **kwargs)

def adarray_sum(self, *args, **kwargs):
    return self.apply(np.ndarray.sum, *args, **kwargs)

ADF.sum = adarray_sum


''' truncates things within numerical precision of 0, but keeps the derivatives'''
def truncate(x, level=1e-16):
    assert x.x > -level
    if isinstance(x.x, Number):
        x.x = max(x.x,0.0)
    else:
        x.x[x.x < 0.0] = 0.0



''' implements product rule for multiplication-like operations, e.g. matrix/tensor multiplication, convolution'''
def ad_product(prod):
    def f(a,b, *args, **kwargs):
        if not isinstance(a, ADF) and not isinstance(b, ADF):
           return prod(a,b)

        a,b = to_auto_diff(a), to_auto_diff(b)
        x = prod(a.x,b.x, *args, **kwargs)

        variables = _get_variables([a,b])
        if not variables:
            return constant(x)

        lc, qc, cp = _make_derivs_dicts()
        for i,v in enumerate(variables):
            lc[v] = prod(a.d(v), b.x, *args, **kwargs) + prod(a.x, b.d(v),*args,**kwargs)
            if get_order() == 2:
                qc[v] = prod(a.d2(v), b.x, *args, **kwargs ) + 2 * prod(a.d(v), b.d(v), *args, **kwargs) + prod(a.x, b.d2(v), *args, **kwargs)

                for j,u in enumerate(variables):
                    if i < j:
                        cp[(v,u)] = prod(a.d2c(u,v), b.x, *args, **kwargs) + prod(a.d(u), b.d(v), *args, **kwargs) + prod(a.d(v) , b.d(u), *args, **kwargs) + prod(a.x, b.d2c(u,v), *args, **kwargs)
        return ADF(x, lc, qc, cp)
    return f

'''matrix multiplication, tensor multiplication, and convolution (Fourier domain multiplication)'''
dot = ad_product(np.dot)
tensordot = ad_product(np.tensordot)
#fftconvolve = ad_product(scipy.signal.fftconvolve)
fftconvolve = ad_product(np.convolve)
outer = ad_product(np.outer)


def diag(x):
    try:
        return x.apply(np.diag)
    except:
        return np.diag(x)

'''A is a constant sparse matrix.
Returns a function f(t,b) = exp(t*A) \dot b'''
def ad_expm_multiply(A):
    def func(t, b):
        if not isinstance(t, ADF) and not isinstance(b, ADF):
            return expm_multiply(t*A, b)
        t,b = to_auto_diff(t), to_auto_diff(b)
        
        if not isinstance(t.x, Number) and len(t) > 1:
            raise Exception("t must be a scalar")

        At = t.x * A
        x = expm_multiply(At, b.x)

        variables = _get_variables([t,b])
        if not variables:
            return constant(x)

        Ax = A.dot(x)
        lc, qc, cp = _make_derivs_dicts()
        b_derivs = {} # stores expm_multiply(At, b.d(v))
        for i,v in enumerate(variables):
            b_derivs[v] = expm_multiply(At, b.d(v))
            lc[v] = Ax * t.d(v) + b_derivs[v]

        if get_order() == 2:
            for v in variables:
                # replace with A * exp(At) * b.dv
                b_derivs[v] = A.dot(b_derivs[v])

            AAx = A.dot(Ax)
            for i, v in enumerate(variables):
                qc[v] = AAx * t.d(v) * t.d(v) + Ax * t.d2(v) + 2 * t.d(v) * b_derivs[v] + expm_multiply(At, b.d2(v))

                for j,u in enumerate(variables):
                    if i < j:
                        cp[(v,u)] = AAx * t.d(u) * t.d(v) + Ax * t.d2c(u,v) + t.d(u) * b_derivs[v] + t.d(v) * b_derivs[u] + expm_multiply(At, b.d2c(u,v))
        return ADF(x, lc, qc, cp)
    return func
