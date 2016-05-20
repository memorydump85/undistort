#! /usr/bin/python

import sys
import numpy as np


def main():
    np.set_printoptions(precision=4, suppress=True)

    #
    # define and parse command line options
    #
    import optparse
    parser = optparse.OptionParser(
        usage = "usage: %prog [options] image1 image2 ...")

    parser.add_option("-t", "--table",
        dest="table_filename",
        help="Undistortion table file")

    options, args = parser.parse_args()

    import struct
    with open(options.table_filename, 'rb') as f:
        H, W = struct.unpack('<II', f.read(8))
        source_map = np.zeros((2, H, W))
        for y in xrange(H):
            for x in xrange(W):
                source_map[:, y, x] = struct.unpack('<dd', f.read(16))

    #
    # process images
    #
    if len(args) == 0:
        print "  ERR: need at least one image as command-line argument"
        sys.exit(-1)

    from skimage.io import imread, imsave
    from skimage.color import rgb2gray
    from skimage import img_as_ubyte
    from skimage.transform import warp

    for filename in args:
        print filename
        im = imread(filename)
        im = img_as_ubyte(rgb2gray(im))
        im_fixed = img_as_ubyte(warp(im, source_map))

        import os.path
        render_filename = os.path.splitext(filename)[0] + '.fixed.png'
        imsave(render_filename, im_fixed)

if __name__ == '__main__':
    main()