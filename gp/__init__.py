import numpy as np
from numpy.linalg import solve, slogdet
from scipy import optimize

import pyximport; pyximport.install()
from gram_matrix import *



#--------------------------------------
class GaussianProcess(object):
#--------------------------------------

    def __init__(self, train_x, train_t, covf):
        self._train_x = train_x
        self._train_t = train_t
        self._covf = covf
        self._C = None
        self._Cinvt = None
        self.fit_result = None


    def ensure_gram_matrix(self):
        if self._C is not None:
            return

        self._C = self._covf.compute_gram_matrix(self._train_x)
        self._Cinvt = solve(self._C, self._train_t)


    def predict(self, query, cov=False):
        return self.__predict(query) if cov else self.__predict_mean(query)


    def __predict_mean(self, query):
        N = len(self._train_x)
        M = len(query)

        data = np.concatenate((self._train_x, query))

        # TODO: compute only relevant parts of A
        A = self._covf.compute_gram_matrix(data)
        Kt = A[N:,:N]

        self.ensure_gram_matrix()
        y_mean = Kt.dot(self._Cinvt)

        return y_mean


    def __predict(self, query):
        N = len(self._train_x)
        M = len(query)

        data = np.concatenate((self._train_x, query))

        # TODO: compute only relevant parts of A
        A = self._covf.compute_gram_matrix(data)
        Kt = A[N:,:N]
        Cq = A[N:,N:]

        self.ensure_gram_matrix()
        y_mean = Kt.dot(self._Cinvt)
        y_cov  = Cq - Kt.dot(solve(self._C, Kt.T))

        return (y_mean, y_cov)


    def model_evidence(self):
        self.ensure_gram_matrix()
        t = self._train_t

        datafit = t.T.dot(self._Cinvt)
        s, logdet = slogdet(self._C)
        complexity = s*logdet
        nomalization = len(t)*np.log(np.pi*2)

        return -0.5 * (datafit + complexity + nomalization)


    @classmethod
    def fit(cls, x, t, covf, theta0):
        evidence = lambda theta: \
            -cls(x, t, covf(theta)).model_evidence()

        if False:
            options = { 'xtol': 0.0001, 'ftol': 0.0001 }
            fit_result = optimize.minimize(evidence, x0=theta0, method='Powell', options=options)
            fit_result.x0 = theta0
        else:
            options = { 'gtol': 1e-05, 'norm': 2 }
            fit_result = optimize.minimize(evidence, x0=theta0, method='CG', options=options)
            fit_result.x0 = theta0

        theta_opt = fit_result.x
        new_gp = cls(x, t, covf(theta_opt))
        new_gp.fit_result = fit_result
        return new_gp


#--------------------------------------
class sqexp1D_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta

    def compute_gram_matrix(self, data):
        return gram_matrix_sq_exp_1D(data, *self.theta)


#--------------------------------------
class sqexp2D_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta

    def compute_gram_matrix(self, data):
        return gram_matrix_sq_exp_2D(data, *self.theta)


#--------------------------------------
class sqexp3D_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta

    def compute_gram_matrix(self, data):
        return gram_matrix_sq_exp_3D(data, *self.theta)


#--------------------------------------
class linear_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta

    def compute_gram_matrix(self, data):
        betaInvI = np.identity(len(data)) / (self.theta[1]**2)
        return self.theta[1]**2 * data.T.dot(data) + betaInvI


#--------------------------------------
class poly_covariancef(object):
#--------------------------------------
    def __init__(self, theta):
        self.theta = theta

    def compute_gram_matrix(self, data):
        betaInvI = np.identity(len(data)) / (self.theta[1]**2)
        return ( data.T.dot(data) + self.theta[0] )**2 + betaInvI
