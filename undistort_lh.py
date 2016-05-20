#! /usr/bin/python

"""
undistort_lh.py

Undistort a tag mosaic using locally weighted homographies.
For a description of the technique please refer:

    Locally-weighted Homographies for Calibration of Imaging Systems
    Pradeep Ranganathan & Edwin Olson

    Proceedings of the IEEE/RSJ International Conference on Intelligent
    Robots and Systems (IROS), October, 2014
"""


import sys
import numpy as np
from math import sqrt



def create_local_homography_object(bandwidth, magnitude, lambda_):
    """
    Helper function for creating WeightedLocalHomography objects
    """
    from mathx.projective_math import WeightedLocalHomography, SqExpWeightingFunction

    H = WeightedLocalHomography(SqExpWeightingFunction(bandwidth, magnitude))
    H.regularization_lambda = lambda_
    return H


def local_homography_error(theta, t_src, t_tgt, v_src, v_tgt):
    """
    This is the objective function used for optimizing parameters of
    the `SqExpWeightingFunction` used for local homography fitting

    Parameters:
    -----------
       `theta` = [ `bandwidth`, `magnitude`, `lambda_` ]:
            parameters of the `SqExpWeightingFunction`

    Arguments:
    -----------
        `t_src`: list of training source points
        `t_tgt`: list of training target points
        `v_src`: list of validation source points
        `v_tgt`: list of validation target points
    """
    H = create_local_homography_object(*theta)
    for s, t in zip(t_src, t_tgt):
        H.add_correspondence(s, t)

    v_mapped = np.array([ H.map(s)[:2] for s in v_src ])
    return ((v_mapped - v_tgt)**2).sum(axis=1).mean()


#--------------------------------------
class GPModel(object):
#--------------------------------------
    def __init__(self, points_i, values):
        assert len(points_i) == len(values)

        X = points_i
        S = np.cov(X.T)

        meanV = np.mean(values, axis=0)
        V = values - np.tile(meanV, (len(values), 1))

        self._meanV = meanV
        self._gp_x = GPModel._fit_gp(X, S, V[:,0])
        self._gp_y = GPModel._fit_gp(X, S, V[:,1])


    @staticmethod
    def _fit_gp(X, covX, t):
        xx, xy, yy = covX[0,0], covX[0,1], covX[1,1]

        # Perform hyper-parameter optimization with different
        # initial points and choose the GP with best model evidence
        theta0 = np.array(( t.std(), sqrt(xx), sqrt(yy), xy, .01 ))

        from gp import GaussianProcess, sqexp2D_covariancef
        best_gp = GaussianProcess.fit(X, t, sqexp2D_covariancef, theta0)

        from sklearn.grid_search import ParameterSampler
        from scipy.stats import uniform
        grid = {
            'sigmaf': uniform(0.1*t.std(), 10*t.std()),
            'cov_xx': uniform(50, 2000),
            'cov_yy': uniform(50, 2000),
            'noise_prec': uniform(0.1, 10)
        }

        for i, sample in enumerate(ParameterSampler(grid, n_iter=100)):
            sys.stdout.write( 'iter: %d/100 evidence: %.4f\r' % (i, best_gp.model_evidence()) )
            sys.stdout.flush()

            theta0 = np.array(( sample['sigmaf'], sample['cov_xx'], sample['cov_yy'], 0, sample['noise_prec']))
            gp = GaussianProcess.fit(X, t, sqexp2D_covariancef, theta0)
            if gp.model_evidence() > best_gp.model_evidence():
                best_gp = gp

        return best_gp


    def predict(self, X):
        V = np.vstack([ self._gp_x.predict(X), self._gp_y.predict(X) ]).T
        return V + np.tile(self._meanV, (len(X), 1))


def process(filename, options):
    #
    # Conventions:
    # a_i, b_i
    #    are variables in image space, units are pixels
    # a_w, b_w
    #    are variables in world space, units are meters
    #
    print '\n========================================'
    print '  File: ' + filename
    print '========================================\n'

    from skimage.io import imread
    im = imread(filename)

    from skimage.color import rgb2gray
    im = rgb2gray(im)

    from skimage import img_as_ubyte
    im = img_as_ubyte(im)

    from apriltag import AprilTagDetector
    detections = AprilTagDetector().detect(im)
    print '  %d tags detected.' % len(detections)

    #
    # Sort detections by distance to center
    #
    c_i = np.array([im.shape[1], im.shape[0]]) / 2.
    dist = lambda p_i: np.linalg.norm(p_i - c_i)
    closer_to_center = lambda d1, d2: int(dist(d1.c) - dist(d2.c))
    detections.sort(cmp=closer_to_center)

    from util.tag36h11_mosaic import TagMosaic
    tag_mosaic = TagMosaic(0.0254)
    mosaic_pos = lambda det: tag_mosaic.get_position_meters(det.id)

    det_i = np.array([ d.c for d in detections ])
    det_w = np.array([ mosaic_pos(d) for d in detections ])

    #
    # To learn a weighted local homography, we find the weighting
    # function parameters that minimize reprojection error across
    # leave-one-out validation folds of the data. Since the
    # homography is local at the center, we only use 5 nearest
    # detections to the center
    #
    det_i5 = det_i[:5]
    det_w5 = det_w[:5]

    from sklearn.cross_validation import LeaveOneOut
    from scipy.optimize import minimize

    def local_homography_loocv_error(theta, args):
        src, tgt = args
        errs = [ local_homography_error(theta, src[t_ix], tgt[t_ix], src[v_ix], tgt[v_ix])
                    for t_ix, v_ix in LeaveOneOut(len(src)) ]
        return np.mean(errs)

    def learn_homography_i2w():
        result = minimize( local_homography_loocv_error,
                    x0=[ 50, 1, 1e-3 ],
                    args=[ det_i5, det_w5 ],
                    method='Powell',
                    options={'ftol': 1e-3} )

        print '\nHomography: i->w'
        print '------------------'
        print '  params:', result.x
        print '    rmse: %.6f' % sqrt(result.fun)
        print '\n  Optimization detail:'
        print '  ' + str(result).replace('\n', '\n      ')

        H = create_local_homography_object(*result.x)
        for i, w in zip(det_i5, det_w5):
            H.add_correspondence(i, w)

        return H

    def learn_homography_w2i():
        result = minimize( local_homography_loocv_error,
                    x0=[ 0.0254, 1, 1e-3 ],
                    method='Powell',
                    args=[ det_w5, det_i5 ],
                    options={'ftol': 1e-3} )

        print '\nHomography: w->i'
        print '------------------'
        print '  params:', result.x
        print '    rmse: %.6f' % sqrt(result.fun)
        print '\n  Optimization detail:'
        print '  ' + str(result).replace('\n', '\n      ')

        H = create_local_homography_object(*result.x)
        for w, i in zip(det_w5, det_i5):
            H.add_correspondence(w, i)

        return H

    #
    # We assume that the distortion is zero at the center of
    # the image and we are interesting in the word to image
    # homography at the center of the image. However, we don't
    # know the center of the image in world coordinates.
    # So we follow a procedure as explained below:
    #
    # First, we learn a homography from image to world
    # Next, we find where the image center `c_i` maps to in
    # world coordinates (`c_w`). Finally, we find the local
    # homography `LH0` from world to image at `c_w`
    #
    H_iw = learn_homography_i2w()
    c_i = np.array([im.shape[1], im.shape[0]]) / 2.
    c_w = H_iw.map(c_i)[:2]
    H_wi = learn_homography_w2i()
    LH0 = H_wi.get_homography_at(c_w)

    print '\nHomography at center'
    print '----------------------'
    print '      c_w =', c_w
    print '      c_i =', c_i
    print 'LH0 * c_w =', H_wi.map(c_w)

    #
    # Obtain distortion estimate
    #       mapped + distortion = detected
    #  (or) distortion = detected - mapped
    #
    def homogeneous_coords(arr):
        return np.hstack([ arr, np.ones((len(arr), 1)) ])

    mapped_i = LH0.dot(homogeneous_coords(det_w).T).T
    mapped_i = np.array([ p / p[2] for p in mapped_i ])
    mapped_i = mapped_i[:,:2]

    distortion = det_i - mapped_i # image + distortion = mapped
    max_distortion = np.max([np.linalg.norm(u) for u in distortion])
    print '\nMaximum distortion is %.2f pixels' % max_distortion

    #
    # Fit non-parametric model to the observations
    #
    model = GPModel(mapped_i, distortion)

    print '\nGP Hyper-parameters'
    print '---------------------'
    print '  x: ', model._gp_x._covf.theta
    print '        log-likelihood: %.4f' % model._gp_x.model_evidence()
    print '  y: ', model._gp_y._covf.theta
    print '        log-likelihood: %.4f' % model._gp_y.model_evidence()
    print ''
    print '  Optimization detail:'
    print '  [ x ]'
    print '  ' + str(model._gp_x.fit_result).replace('\n', '\n      ')
    print '  [ y ]'
    print '  ' + str(model._gp_y.fit_result).replace('\n', '\n      ')

    #
    # Use the non-parametric model to compute the source map
    # we compute the map in blocks to reduce our memory requirements
    #
    print '\nComputing source map'
    print '----------------------'
    H, W = im.shape
    center = np.array([W/2., H/2.])

    def scale_coords(xy):
        return (xy - center)*float(options.output_scale) + center

    source_map = np.zeros((2, H, W))

    B = 50 # Block size is 50x50
    for sy in xrange(0, H, B):
        sys.stdout.write('  ')

        for sx in xrange(0, W, B):
            ty, tx = min(sy+B, H), min(sx+B, W)
            dst_coords = np.array([[x, y] for y in xrange(sy, ty) for x in xrange(sx, tx)])
            dst_coords = scale_coords(dst_coords)
            src_coords = dst_coords + model.predict(dst_coords)
            source_map[1, sy:ty, sx:tx] = src_coords[:,0].reshape((ty-sy, tx-sx))
            source_map[0, sy:ty, sx:tx] = src_coords[:,1].reshape((ty-sy, tx-sx))

            sys.stdout.write('. ')
            sys.stdout.flush()
        sys.stdout.write('\n')

    #
    # output undistortion table if requested
    #
    if not options.no_write_table:
        table_filename = filename + '.table'
        print '\nWriting undistortion table to %s' % table_filename

        import struct
        with open(table_filename, 'wb') as f:
            f.write( struct.pack('<II', H, W) )
            for y in xrange(H):
                for x in xrange(W):
                    f.write( struct.pack('<dd', *source_map[:, y, x]) )

    #
    # undistorted image if requested
    #
    if not options.no_write_image:
        import os.path
        render_filename = os.path.splitext(filename)[0] + '.fixed.png'
        print '\nRendering undistorted image: %s' % render_filename

        from skimage.transform import warp
        im_fixed = img_as_ubyte(warp(im, source_map))

        from skimage.io import imsave
        imsave(render_filename, im_fixed)


def main():
    np.set_printoptions(precision=4, suppress=True)

    #
    # define and parse command line options
    #
    import optparse
    parser = optparse.OptionParser(
        usage = "usage: %prog [options] image1 image2 ...")

    parser.add_option("", "--no-table",
        dest="no_write_table", action="store_true", default=False,
        help="Do not write out undistortion source mapping table")

    parser.add_option("", "--no-image",
        dest="no_write_image", action="store_true", default=False,
        help="Do not render undistorted image")

    parser.add_option("-s", "--scale",
        dest="output_scale", default=1.0,
        help="Scale factor for output image. defaults to 1.0. The output " +
             "image can be larger than the input image; use this " + 
             "option to control output image cropping.")

    options, args = parser.parse_args()

    #
    # process images
    #
    if len(args) == 0:
        print "  ERR: need at least one image as command-line argument"
        sys.exit(-1)

    for filename in args:
        process(filename, options)

if __name__ == '__main__':
    main()