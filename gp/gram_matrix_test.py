import sys
import numpy as np
from scipy.sparse.linalg import cg
from time import time

import pyximport; pyximport.install()
import gram_matrix


#--------------------------------------
class sqexp3D_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta
        self.fsig, sig00, sig11, sig22, var10, var20, var21, self.noise_prec = theta
        Sigma = np.array([ [ sig00**2, var10,     var20   ],
                           [ var10,    sig11**2,  var21   ],
                           [ var20,    var21,     sig22**2] ])
        self.Sigma_inv = np.linalg.inv(Sigma)

    def __call__(self, a, b, colocated):
        # chi2 = zT * Sigma_inv * z
        z = a - b
        chi2 = z.dot(self.Sigma_inv).dot(z)

        nu = self.fsig
        beta = self.noise_prec

        v = (nu*nu)*np.exp(-0.5*chi2)
        return v + 1./(beta*beta) if colocated else v


def py_gram_matrix_sq_exp_3D(data,
    sigma_f, sigma_xx, sigma_yy, sigma_zz,
    corr_xy, corr_yz, corr_xz, noise_precision):
    covf = sqexp3D_covariancef(
            [sigma_f, sigma_xx, sigma_yy, sigma_zz, corr_xy, corr_yz, corr_xz, noise_precision])

    N = len(data)
    C = np.array([
            covf(data[i], data[j], colocated=(i==j))
            for i in xrange(0, N) for j in xrange (0, N)
                ]).reshape(N,N)

    return C


#--------------------------------------
class sqexp2D_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta
        self.fsig, sig00, sig11, var10, self.noise_prec = theta
        Sigma = np.array([[sig00**2, var10], [var10, sig11**2]])
        self.Sigma_inv = np.linalg.inv(Sigma)

    def __call__(self, a, b, colocated):
        # chi2 = zT * Sigma_inv * z
        z = a - b
        chi2 = z.dot(self.Sigma_inv).dot(z)

        nu = self.fsig
        beta = self.noise_prec

        v = (nu*nu)*np.exp(-0.5*chi2)
        return v + 1./(beta*beta) if colocated else v


def py_gram_matrix_sq_exp_2D(data,
    sigma_f, sigma_xx, sigma_yy, corr_xy, noise_precision):
    covf = sqexp2D_covariancef(
            [sigma_f, sigma_xx, sigma_yy, corr_xy, noise_precision])

    N = len(data)
    C = np.array([
            covf(data[i], data[j], colocated=(i==j))
            for i in xrange(0, N) for j in xrange (0, N)
                ]).reshape(N,N)

    return C


#--------------------------------------
class sqexp1D_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta
        self.fsig, self.sig, self.noise_prec = theta

    def __call__(self, a, b, colocated):
        z = a - b
        v = (self.fsig*self.fsig)* np.exp(-0.5*(z*z)/(self.sig*self.sig))
        return v + 1./(self.noise_prec*self.noise_prec) if colocated else v


def py_gram_matrix_sq_exp_1D(data,
    sigma_f, sigma_x, noise_precision):
    covf = sqexp1D_covariancef(
            [sigma_f, sigma_x, noise_precision])

    N = len(data)
    C = np.array([
            covf(data[i], data[j], colocated=(i==j))
            for i in xrange(0, N) for j in xrange (0, N)
                ]).reshape(N,N)

    return C


np.set_printoptions(precision=4, suppress=True)


print '\n--3D--------------------\n'
data = np.random.randn(800,3)

t0 = time()
pyG = py_gram_matrix_sq_exp_3D(data, 1, 1, 1, 1, 0, 0, 0, 10)
print '  py: %.4fs' % (time()-t0)

t0 = time()
cyG = gram_matrix.gram_matrix_sq_exp_3D(data, 1, 1, 1, 1, 0, 0, 0, 10)
print '  cy: %.4fs' % (time()-t0)

# print 'py:\n', pyG
# print 'cy:\n', cyG

assert np.allclose(pyG, cyG)


print '\n--2D--------------------\n'
data = np.random.randn(800,2)

t0 = time()
pyG = py_gram_matrix_sq_exp_2D(data, 1, 1, 1, 0, 10)
print '  py: %.4fs' % (time()-t0)

t0 = time()
cyG = gram_matrix.gram_matrix_sq_exp_2D(data, 1, 1, 1, 0, 10)
print '  cy: %.4fs' % (time()-t0)

# print 'py:\n', pyG
# print 'cy:\n', cyG

assert np.allclose(pyG, cyG)


print '\n--1D--------------------\n'
data = np.random.randn(800)

t0 = time()
pyG = py_gram_matrix_sq_exp_1D(data, 1, 1, 10)
print '  py: %.4fs' % (time()-t0)

t0 = time()
cyG = gram_matrix.gram_matrix_sq_exp_1D(data, 1, 1, 10)
print '  cy: %.4fs' % (time()-t0)

# print 'py:\n', pyG
# print 'cy:\n', cyG

assert np.allclose(pyG, cyG)