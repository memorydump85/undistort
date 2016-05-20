import numpy as np
from tupletypes import Correspondence



class UnitWeightingFunction(object):
    def __call__(self, p, q):
        return 1


class SqExpWeightingFunction(object):
    def __init__(self, bandwidth, magnitude=1.):
        self._tau = bandwidth
        self._nu = magnitude

    def __call__(self, p, q):
        z = np.subtract(p, q) / self._tau
        return self._nu*self._nu * np.exp(-(z*z).sum())


def _normalization_transform(points):
    muX, muY = np.mean(points, axis=0)
    stdX, stdY = np.std(points, axis=0)
    scaleX, scaleY = 1./stdX, 1./stdY

    X = np.array([
            [ scaleX,     0, -muX*scaleX ],
            [ 0,     scaleY, -muY*scaleY ],
            [ 0,          0,           1 ]])

    Xinv = np.array([
            [ stdX,       0,         muX ],
            [ 0,       stdY,         muY ],
            [ 0,          0,           1 ]])

    return X, Xinv


def _homogeneous_coords(p):
    assert len(p)==2 or len(p)==3
    return p if len(p)==3 else np.array([p[0], p[1], 1.])


#--------------------------------------
class WeightedLocalHomography(object):
#--------------------------------------
    def __init__(self, wfunc=UnitWeightingFunction()):
        self._corrs = []
        self.regularization_lambda = 0
        self._weighting_func = wfunc
        self._precompute_done = False


    def add_correspondence(self, source_xy, target_xy):
        assert len(source_xy) == 2
        assert len(target_xy) == 2
        self._corrs.append(Correspondence(source_xy, target_xy))


    def _precompute(self):
        """ Precompute normalization transforms and constraint
        matrix. These remain the same for every homography query """
        if self._precompute_done == True:
            return

        self._srcX, _             = _normalization_transform([ c.source for c in self._corrs ])
        self._tgtX, self._tgtXinv = _normalization_transform([ c.target for c in self._corrs ])

        constraints = []
        for c in self._corrs:
            x, y, _ = self._srcX.dot(_homogeneous_coords(c.source))
            i, j, _ = self._tgtX.dot(_homogeneous_coords(c.target))

            constraints.append( [-x, -y, -1, 0, 0, 0, i*x, i*y, i] )
            constraints.append( [0, 0, 0, -x, -y, -1, j*x, j*y, j] )

        self.constraint_matrix = np.array(constraints, dtype=np.float)
        self._precompute_done = True


    def regularized_weighting_function(self, p, q):
        return self._weighting_func(p, q) + self.regularization_lambda**2


    def get_correspondence_weights(self, src_pt):
        """ How similar is `src_pt` to each source point in `self._corrs` """
        return [ self.regularized_weighting_function(src_pt, c.source) for c in self._corrs ]


    def get_homography_at(self, src_pt):
        self._precompute()

        A = self.constraint_matrix

        # The weighting is a diagonal matrix that encodes how similar
        # `src_pt` is to each source point in `self._corrs`
        w_diag = self.get_correspondence_weights(src_pt)

        # Each correspondence produces 2 constraints. So weights also
        # need to be repeated
        W = np.diag(np.sqrt(np.repeat(w_diag, 2)))
        assert len(A) == len(W)

        # Homography is the total least squares solution: The eigen-vector
        # corresponding to the smallest eigen-value
        U, s, Vt = np.linalg.svd(W.dot(A))
        H = Vt.T[:,-1].reshape((3,3))

        return reduce(np.dot, [self._tgtXinv, H, self._srcX])


    def map(self, src_pt):
        """ Map `src_pt` to the target plane using the local
        homography at `src_pt` """
        m = self.get_homography_at(src_pt).dot(_homogeneous_coords(src_pt))
        m /= m[2]
        return m
