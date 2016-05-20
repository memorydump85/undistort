import sys
import numpy as np
from skimage.io import imread
from skimage.util import img_as_ubyte
from apriltag import AprilTagDetector


def main():
    im = imread(sys.argv[1])
    im = img_as_ubyte(im)

    tagdetector = AprilTagDetector()
    print "\n".join([ str((d.id, d.c)) for d in tagdetector.detect(im) ])

if __name__ == '__main__':
    main()