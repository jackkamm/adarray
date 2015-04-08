from .ad import *
from .ad import ADF, _get_variables, check_auto_diff

import numpy as np
from numbers import Number
import scipy, scipy.signal

def array(x):
    if isinstance(x, ADF):
        return x.apply(np.array)

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if np.issubdtype(x.dtype, np.number):
        return ADF(x,{},{},{})

    # get the variables to differentiate against
    adentries = []
    for i,xi in np.ndenumerate(x):
        if isinstance(xi, ADF):
            adentries.append((i,xi))
        elif not isinstance(xi,Number):
            raise TypeError(str((i,xi)))
    variables = _get_variables([xi for _,xi in adentries])

    # initialize the dictionaries of derivatives
    lc_dict, qc_dict, cp_dict = {}, {}, {}
    if variables:
        d_dicts = (lc_dict, qc_dict, cp_dict)

        # fill the dictionaries of derivatives
        for i,xi in adentries:
            for xi_d, x_d in zip((xi._lc, xi._qc, xi._cp), d_dicts):
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

    return ADF(x, lc_dict, qc_dict, cp_dict)


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
    lc, qc, cp = {},{},{}
    for ret_d, x_d in zip((lc, qc, cp), (self._lc, self._qc, self._cp)):
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

''' implements product rule for multiplication-like operations, e.g. matrix/tensor multiplication, convolution'''
def ad_product(prod):
    def f(a,b, *args, **kwargs):
        #if not isinstance(a, ADF) and not isinstance(b, ADF):
        #    return prod(a,b)

        a,b = check_auto_diff(a), check_auto_diff(b)
        x = prod(a.x,b.x, *args, **kwargs)

        variables = _get_variables([a,b])
        if not variables:
            return ADF(x, {}, {}, {})

        lc, qc, cp = {}, {}, {}
        for i,v in enumerate(variables):
            lc[v] = prod(a.d(v), b.x, *args, **kwargs) + prod(a.x, b.d(v),*args,**kwargs)
            qc[v] = prod(a.d2(v), b.x, *args, **kwargs ) + 2 * prod(a.d(v), b.d(v), *args, **kwargs) + prod(a.x, b.d2(v), *args, **kwargs)

            for j,u in enumerate(variables):
                if i < j:
                    cp[(v,u)] = prod(a.d2c(u,v), b.x, *args, **kwargs) + prod(a.d(u), b.d(v), *args, **kwargs) + prod(a.d(v) , b.d(u), *args, **kwargs) + prod(a.x, b.d2c(u,v), *args, **kwargs)
        return ADF(x, lc, qc, cp)
    return f

'''matrix multiplication, tensor multiplication, and convolution (Fourier domain multiplication)'''
dot = ad_product(np.dot)
tensordot = ad_product(np.tensordot)
fftconvolve = ad_product(scipy.signal.fftconvolve)
