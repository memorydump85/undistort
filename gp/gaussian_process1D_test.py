import numpy as np
from matplotlib import pyplot
from gp import *


if __name__ == "__main__":
#--------------------------------------
    x = (np.random.random(100))*2
    t = np.array(map(lambda a: a**2 + 0.05*np.random.randn(), x))
    t = t - np.mean(t)

    # Find the best set of hyper-parameters
    # theta0 = [np.std(t), 1, 10]
    # gp = GaussianProcess.fit(x, t, sqexp1D_covariancef, theta0)

    theta0 = [1, 1]
    gp = GaussianProcess.fit(x, t, poly_covariancef, theta0)
    print gp._covf.theta

    # Plot predictions and samples from the Gaussian Process
    q = np.arange(-1.2, 1.2, 0.051)
    mean, cov = gp.predict(q, cov=True)
    assert (mean == gp.predict(q)).all()

    sig_bnd = np.sqrt(np.diag(cov))

    for s in np.random.multivariate_normal(mean, cov, 5):
        pyplot.plot(q, s, 'y-')

    pyplot.plot(x, t, '.')
    pyplot.plot(q, mean, 'r-')
    pyplot.plot(q, mean + 2*sig_bnd, 'k-')
    pyplot.plot(q, mean - 2*sig_bnd, 'k-')
    pyplot.show(block=True)
