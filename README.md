

undistort
=========

Computes an undistortion table using locally-weighted homographies.
The paper decsribing this method is available:

[_Locally-weighted Homographies for Calibration of Imaging Systems_](http://april.eecs.umich.edu/papers/details.php?name=ranganathan2014iros)<br/>
Pradeep Ranganathan and Edwin Olson


Required python packages
------------------------
`numpy >= 1.9.0` <br/>
`scipy >= 0.16.1` <br/>
`scikit-learn >= 0.16.1` <br/>
`scikit-image >= 0.11.3` <br/>
`cython >= 0.22.1`

How To
-------

Install the required dependencies:
```bash
pip install numpy scipy scikit-learn scikit-image cython
```

Run the `undistort.py` script. Example:
```bash
./undistort.py --scale=1.5 examples/example.png
```

This runs the undistortion script on the `examples/example.png` and
produces the undistortion table `examples/example.png.table` and
rectified image `examples/example.fixed.png`.

The produced `.table` file can be used to undistort other images using
the `examples/render_undistorted.py` script.

Notes
-----

The file `examples/example.png` is an example of an ideal image for
estimating distortion. It provides features that can cover the entire
image. The non-parametric nature of the distortion model makes it
data-intensive -- the model is accurate near observed data points. Hence
an image with detections that covers the entire image produces better
models.

The file `examples/bad_example.png` is an example of an image that might
not produce a good distortion model.

The `undistort.py` script also accepts multiple input images on the
command line. In this case it computes a distortion model using
tag observations from all the input images.
