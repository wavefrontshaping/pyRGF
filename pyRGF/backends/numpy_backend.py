import numpy as np
# import scipy

arcsin = np.arcsin
sin = np.sin
cos = np.cos
exp = np.exp
sqrt = np.sqrt
arange = np.arange
real = np.real
imag = np.imag
transpose = np.transpose
eye = np.eye
diag = np.diag
inv = np.linalg.inv
matmul = np.matmul
mul = np.multiply
conjugate = np.conjugate

real_to_complex = lambda x: x

def real(A):
    return A.real

def imag(A):
    return A.imag

def to_complex(A,B):
    return A+1j*B


def rand_real(s):
    return np.random.rand(*s)

def ones(shape, dtype=complex, name=None):
    return np.ones(shape, dtype=dtype)

def zeros(shape, dtype=complex, name=None):
    return np.zeros(shape, dtype=dtype)

# def round_int(x):
#     return int(np.round(x))

# def floor_int(x):
#     return int(np.floor(x))

def to_float32(x):
    return np.float32(x)

def is_shape(A, s):
    return (np.array(A.shape) != s).all()

