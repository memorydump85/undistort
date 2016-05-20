import cython
import numpy as np

cimport cpython
cimport numpy as np
from cpython.cobject cimport PyCObject_FromVoidPtr, PyCObject_AsVoidPtr


#
# Cython declarations of external C functions
#

#--------------------------------------
cdef extern from "zarray.h":
#--------------------------------------
    ctypedef struct zarray_t:
        pass

    inline int zarray_size(const zarray_t *za)
    inline void zarray_get(const zarray_t *za, int idx, void *p)


#--------------------------------------
cdef extern from "image_u8.h":
#--------------------------------------
    ctypedef struct image_u8_t:
        int width
        int height
        int stride
        np.uint8_t *buf

    image_u8_t *image_u8_create(unsigned int width, unsigned int height)
    void image_u8_destroy(image_u8_t *im)


#--------------------------------------
cdef extern from "matd.h":
#--------------------------------------
    ctypedef struct matd_t:
        pass

    double MATD_EL(void* name, int row, int col)
    double matd_get(const matd_t *m, int row, int col)


#--------------------------------------
cdef extern from "apriltag.h":
#--------------------------------------
    ctypedef struct apriltag_family_t:
        np.uint32_t black_border

    ctypedef struct apriltag_detector_t:
        int nthreads;
        float quad_decimate;
        float quad_sigma;
        int refine_edges;
        int refine_decode;
        int refine_pose;
        int debug;

    ctypedef struct apriltag_detection_t:
        apriltag_family_t *family;
        int id;
        int hamming;
        float goodness;
        float decision_margin;
        matd_t *H;
        double c[2];
        double p[4][2];

    apriltag_detector_t *apriltag_detector_create()
    void apriltag_detector_add_family(apriltag_detector_t *td, apriltag_family_t *fam)
    void apriltag_detector_destroy(apriltag_detector_t *td)
    zarray_t *apriltag_detector_detect(apriltag_detector_t *td, image_u8_t *im_orig)
    void apriltag_detections_destroy(zarray_t *detections)


#--------------------------------------
cdef extern from "tag36h11.h":
#--------------------------------------
    apriltag_family_t *tag36h11_create()
    void tag36h11_destroy(apriltag_family_t *tf)


#--------------------------------------
cdef extern from "tag36h10.h":
#--------------------------------------
    apriltag_family_t *tag36h10_create()
    void tag36h10_destroy(apriltag_family_t *tf)


#--------------------------------------
cdef extern from "tag36artoolkit.h":
#--------------------------------------
    apriltag_family_t *tag36artoolkit_create()
    void tag36artoolkit_destroy(apriltag_family_t *tf)


#--------------------------------------
cdef extern from "tag25h9.h":
#--------------------------------------
    apriltag_family_t *tag25h9_create()
    void tag25h9_destroy(apriltag_family_t *tf)


#--------------------------------------
cdef extern from "tag25h7.h":
#--------------------------------------
    apriltag_family_t *tag25h7_create()
    void tag25h7_destroy(apriltag_family_t *tf)


#
# Cython glue for the april tag detector
#

from collections import namedtuple
AprilTagDetection = namedtuple('AprilTagDetection',
                        ['id', 'hamming', 'goodness', 'decision_margin', 'H', 'c', 'p'])


cdef create_AprilTagDetection_from_struct(apriltag_detection_t *det):
    """
    Convert from a apriltag_detection_t (c-side) to a AprilTagDetection (python)
    """
    id_             = det.id
    hamming         = det.hamming
    goodness        = det.goodness
    decision_margin = det.decision_margin
    H               = np.array([
                            MATD_EL(det.H,0,0), MATD_EL(det.H,0,1), MATD_EL(det.H,0,2),
                            MATD_EL(det.H,1,0), MATD_EL(det.H,1,1), MATD_EL(det.H,1,2),
                            MATD_EL(det.H,2,0), MATD_EL(det.H,2,1), MATD_EL(det.H,2,2),
                      ]).reshape((3,3))
    c               = np.array(det.c)
    p               = np.array(det.p)

    return AprilTagDetection(id_, hamming, goodness, decision_margin, H, c, p)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef image_u8_t *image_u8_create_from_ndarray(np.ndarray[np.uint8_t, ndim=2, mode='c'] arr):
    cdef image_u8_t* im
    im = image_u8_create(arr.shape[1], arr.shape[0])

    cdef int y, x
    for y in xrange(im.height):
        for x in xrange(im.width):
            im.buf[im.stride*y+x] = arr[y,x]

    return im


#--------------------------------------
class AprilTagDetector(object):
#--------------------------------------
    def __init__(self, tagfamily='tag36h11', debug=False,
                 border_size=1, n_threads=1, decimate=1., blur_sigma=0.,
                 refine_edges=1, refine_decode=0, refine_pose=0):
        self.tagfamily = tagfamily

        cdef apriltag_family_t *tf_
        if tagfamily == "tag36h11":
            tf_ = tag36h11_create()
        elif tagfamily == "tag36h10":
            tf_ = tag36h10_create()
        elif tagfamily == "tag36artoolkit":
            tf_ = tag36artoolkit_create()
        elif tagfamily == "tag25h9":
            tf_ = tag25h9_create()
        elif tagfamily == "tag25h7":
            tf_ = tag25h7_create()
        else:
            raise Exception("Unrecognized tag family name: " + tagfamily)

        tf_.black_border = border_size
        self.tf = PyCObject_FromVoidPtr(tf_, NULL)

        cdef apriltag_detector_t *td_
        td_ = apriltag_detector_create()
        td_.quad_decimate = decimate
        td_.quad_sigma = blur_sigma
        td_.nthreads = n_threads
        td_.debug = debug
        td_.refine_edges = refine_edges
        td_.refine_decode = refine_decode
        td_.refine_pose = refine_pose
        self.td = PyCObject_FromVoidPtr(td_, NULL)

        apriltag_detector_add_family(td_, tf_)


    def __del__(self):
        apriltag_detector_destroy(<apriltag_detector_t *>PyCObject_AsVoidPtr(self.td))

        cdef apriltag_family_t *tf_
        tf_ = <apriltag_family_t*>PyCObject_AsVoidPtr(self.tf)
        if self.tagfamily == "tag36h11":
            tag36h11_destroy(tf_)
        elif self.tagfamily == "tag36h10":
            tag36h10_destroy(tf_)
        elif self.tagfamily == "tag36artoolkit":
            tag36artoolkit_destroy(tf_)
        elif self.tagfamily == "tag25h9":
            tag25h9_destroy(tf_)
        elif self.tagfamily == "tag25h7":
            tag25h7_destroy(tf_)


    def detect(self, np.ndarray[np.uint8_t, ndim=2, mode='c'] im):
        cdef image_u8_t* im_u8
        im_u8 = image_u8_create_from_ndarray(im)

        cdef apriltag_detector_t *td_
        td_ = <apriltag_detector_t*>PyCObject_AsVoidPtr(self.td)

        cdef zarray_t *c_detections
        c_detections = apriltag_detector_detect(td_, im_u8)
        image_u8_destroy(im_u8)

        cdef apriltag_detection_t *det
        py_detections = []
        for i in xrange(zarray_size(c_detections)):
            zarray_get(c_detections, i, cython.address(det))
            py_detections.append(create_AprilTagDetection_from_struct(det))

        apriltag_detections_destroy(c_detections)
        return py_detections
