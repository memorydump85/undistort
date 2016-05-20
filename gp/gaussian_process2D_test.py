import numpy as np
from mayavi import mlab
from gp import GaussianProcess, sqexp2D_covariancef


if __name__ == "__main__":
#--------------------------------------
    # sample training data
    x = np.array([np.array([i,j]) for i in np.arange(-1, 1, 0.2) for j in np.arange(-1, 1, 0.2)])
    t = np.array([r[0]**2+r[1]**2 + r[0]*r[1] + 0.1*np.random.randn() for r in x])
    t = t - np.mean(t)

    # Find the best set of hyper-parameters
    theta0 = [np.std(t), 1, 1, 0, 10]
    gp = GaussianProcess.fit(x, t, sqexp2D_covariancef, theta0)
    print gp.covf.theta

    # Plot predictions and samples from the Gaussian Process
    q = np.array([np.array([i,j]) for i in np.arange(-1.1, 1.1, 0.2) for j in np.arange(-1.1, 1.1, 0.2)])
    mean, cov = gp.predict(q, cov=True)
    assert (mean == gp.predict(q)).all()

    sig_bnd = np.sqrt(np.diag(cov))

    mlab.points3d(x[:,0], x[:,1], t, t, scale_mode='none', scale_factor=0.01)

    pts = mlab.points3d(q[:,0], q[:,1], mean, mean, scale_mode='none', scale_factor=0.001)
    mesh = mlab.pipeline.delaunay2d(pts)
    surf = mlab.pipeline.surface(mesh, opacity=0.7)

    pts = mlab.points3d(q[:,0], q[:,1], mean+2*sig_bnd, mean+2*sig_bnd, scale_mode='none', scale_factor=0.001)
    mesh = mlab.pipeline.delaunay2d(pts)
    surf = mlab.pipeline.surface(mesh, opacity=0.2)

    pts = mlab.points3d(q[:,0], q[:,1], mean-2*sig_bnd, mean-2*sig_bnd, scale_mode='none', scale_factor=0.001)
    mesh = mlab.pipeline.delaunay2d(pts)
    surf = mlab.pipeline.surface(mesh, opacity=0.2)

    mlab.show()
