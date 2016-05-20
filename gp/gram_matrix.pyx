import numpy as np
import cython

cimport numpy as np
from libc.math cimport exp



ctypedef np.float64_t float64_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gram_matrix_sq_exp_1D(
    np.ndarray[float64_t, ndim=1] data,
    float64_t sigma_f,          # function/signal variance
    float64_t sigma_x,          # length-scale
    float64_t sigma_inv_noise   # sqrt(noise precision)
    ):

    cdef int N
    N = data.shape[0]

    cdef float64_t noise_variance, x_precision
    noise_variance = 1./(sigma_inv_noise*sigma_inv_noise)
    x_precision = 1./(sigma_x*sigma_x)

    cdef int i, j
    cdef float64_t z

    cdef np.ndarray[float64_t, ndim=2] K
    K = np.empty((N, N))

    for i in xrange(N):
        for j in xrange(i+1):
            z = data[i] - data[j]

            v = (sigma_f*sigma_f)*exp(-0.5*z*z*x_precision)
            if i==j: v += noise_variance

            K[i,j] = K[j,i] = v

    return K


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gram_matrix_sq_exp_2D(
    np.ndarray[float64_t, ndim=2] data,
    float64_t sigma_f,          # Function/signal variance
    float64_t sigma_xx,         # 2D length-scale parameters
    float64_t sigma_yy,         #   ..
    float64_t corr_xy,          #   ..
    float64_t sigma_inv_noise   # sqrt(noise precision)
    ):

    cdef np.ndarray[float64_t, ndim=2] Sigma, Sigma_inv
    Sigma = np.array([ [ sigma_xx**2,  corr_xy    ],
                       [  corr_xy,    sigma_yy**2 ] ])
    Sigma_inv = np.linalg.inv(Sigma)

    cdef float64_t p, q, r, s
    p = Sigma_inv[0,0]
    q = Sigma_inv[0,1]
    r = Sigma_inv[1,0]
    s = Sigma_inv[1,1]

    cdef int N
    N = data.shape[0]

    cdef float64_t noise_variance
    noise_variance = 1./(sigma_inv_noise*sigma_inv_noise)

    cdef int i, j
    cdef float64_t g, h, chi2, v

    cdef np.ndarray[float64_t, ndim=2] K
    K = np.empty((N, N))

    for i in xrange(N):
        for j in xrange(i+1):
            g = data[i,0] - data[j,0]
            h = data[i,1] - data[j,1]

            chi2 = g*(p*g+q*h) + h*(r*g+s*h)
            v = (sigma_f*sigma_f)*exp(-0.5*chi2)
            if i==j: v += noise_variance

            K[i,j] = K[j,i] = v

    return K


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gram_matrix_sq_exp_3D(
    np.ndarray[float64_t, ndim=2] data,
    float64_t sigma_f,          # Function/signal variance
    float64_t sigma_xx,         # 3D length-scale parameters
    float64_t sigma_yy,         #   ..
    float64_t sigma_zz,         #   ..
    float64_t corr_xy,          #   ..
    float64_t corr_yz,          #   ..
    float64_t corr_xz,          #   ..
    float64_t sigma_inv_noise   # sqrt(noise precision)
    ):

    cdef np.ndarray[float64_t, ndim=2] Sigma, Sigma_inv
    Sigma = np.array([ [ sigma_xx**2,   corr_xy,     corr_xz   ],
                       [  corr_xy,     sigma_yy**2,  corr_yz   ],
                       [  corr_xz,      corr_yz,    sigma_zz**2 ] ])
    Sigma_inv = np.linalg.inv(Sigma)

    cdef float64_t p, q, r, s, t, u
    p = Sigma_inv[0,0]
    q = Sigma_inv[0,1]
    r = Sigma_inv[1,1]
    s = Sigma_inv[0,2]
    t = Sigma_inv[1,2]
    u = Sigma_inv[2,2]

    cdef int N
    N = data.shape[0]

    cdef float64_t noise_variance
    noise_variance = 1./(sigma_inv_noise*sigma_inv_noise)

    cdef int i, j
    cdef float64_t a, b, c, chi2, v

    cdef np.ndarray[float64_t, ndim=2] K
    K = np.empty((N, N))

    for i in xrange(N):
        for j in xrange(i+1):
            a = data[i,0] - data[j,0]
            b = data[i,1] - data[j,1]
            c = data[i,2] - data[j,2]

            chi2 = a*(p*a + q*b + s*c) + b*(q*a + r*b + t*c) + c*(s*a + t*b + u*c)
            v = (sigma_f*sigma_f)*exp(-0.5*chi2)
            if i==j: v += noise_variance

            K[i,j] = K[j,i] = v

    return K
