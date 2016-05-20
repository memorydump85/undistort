from collections import namedtuple



Correspondence = namedtuple('Correspondence', [
    'source',
    'target'
])


WorldImageHomographyInfo = namedtuple('WorldImageHomographyInfo', [
    'H',                # Weighted local homography object
    'c_w',              # center of image in world coordinates (meters)
    'c_i',              # center of the image in pixel coordinates
])
