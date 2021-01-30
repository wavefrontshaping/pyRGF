import torch

DEFAULT_DTYPE = torch.float32

arcsin = torch.asin
sin = torch.sin
cos = torch.cos
exp = torch.exp
sqrt = torch.sqrt
arange = torch.arange
# inv = torch.inverse

# def to_complex()

# class complex_tensor(torch.tensor):
#     @property
#     def real(self):
#         return self.select(-1, 0)
    
#     @property
#     def imag(self):
#         return self.select(-1, 1)

def real(A):
    return A.select(-1,0)

def real(A):
    return A.select(-1,1)
    

def decor_real_to_complex(func):
    '''
    
    '''
    def inner(*args, **kwargs):
        A = func(*args, **kwargs)
        return torch.stack((A, torch.zeros_like(A)), 
                           dim = -1)
    return inner

eye = decor_real_to_complex(torch.eye)
rand_real = decor_real_to_complex(torch.rand)
ones = decor_real_to_complex(torch.ones)

def real_to_complex(A):
    return torch.stack((A, torch.zeros_like(A)), 
                       dim = -1)

def to_complex(A,B):
    return torch.stack((A, B), 
                       dim = -1)

def diag(A):
    return to_complex(torch.diag(A[...,0].squeeze(-1)),
                      torch.diag(A[...,1].squeeze(-1)))


def inv(A):
    '''
    Invert a complex value tensor.
    '''
    R = A[...,0]
    I = A[...,1]
    return torch.stack((torch.inverse(R+I@torch.inverse(R)@I),
                       -torch.inverse(I+R@torch.inverse(I)@R)),
                       dim = -1)

#      (A + B*A^-1*B)^-1 - i*(B + A*B^-1*A)^-1 


# def eye(s, dtype = None):
#     if not dtype:
#         dtype = DEFAULT_DTYPE
#     return to_complex(torch.eye(s, dtype = dtype))

# def rand_real(s, dtype = None):
#     if not dtype:
#         dtype = DEFAULT_DTYPE
#     return to_complex(torch.rand(s, dtype = dtype))
# #     return torch.stack((torch.rand(s, dtype = dtype), 
# #                         torch.zeros(s, dtype = dtype)), 
#                        dim = -1)

def real(A):
    return A[...,0]

def imag(A):
    return A[...,1]

# def ones(shape, dtype=None, name=None):
#     return torch.ones(shape, dtype=dtype)

def zeros(shape, dtype=None):
    return torch.zeros(shape+[2], dtype=dtype)

def matmul(A,B):
    '''
    Matrix multiplication for complex tensors. 
    Tensors have to have a last dimension of size 2 for real and imaginary parts.
    The -2 and -3 dimensions are the 2 dimensions to multiply.
    Other previous dimensions are considered as batch dimensions (cf PyTorch matmul() function).
    '''
    return torch.stack((A[...,0].matmul(B[...,0])-A[...,1].matmul(B[...,1]),
                    A[...,0].matmul(B[...,1])+A[...,1].matmul(B[...,0])),dim=-1)

def mul(A,B):
    '''
    Element-wise multiplication for complex tensors. 
    Tensors have to have a last dimension of size 2 for real and imaginary parts.
    The -2 and -3 dimensions are the 2 dimensions to multiply.
    Other previous dimensions are considered as batch dimensions (cf PyTorch mul() function).
    '''
    return torch.stack((A[...,0].mul(B[...,0])-A[...,1].mul(B[...,1]),
                    A[...,0].mul(B[...,1])+A[...,1].mul(B[...,0])),dim=-1)

def to_float32(x):
    return x.type(torch.float32)

def transpose(A):
    return torch.transpose(A,-3,-2)

def is_shape(A, s):
    return torch.all(torch.eq(torch.tensor(A.shape[:-1]), torch.tensor(s)))