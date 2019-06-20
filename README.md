undistort
=========

Computes an undistortion table using locally-weighted homographies.
The method used is described in:

> [_Locally-weighted Homographies for Calibration of Imaging Systems_](http://april.eecs.umich.edu/papers/details.php?name=ranganathan2014iros)<br/>
> Pradeep Ranganathan and Edwin Olson


Quickstart with Docker
----------------------

Use the provided `Dockerfile` to build a docker image and run this code
inside it.
```bash
docker build -t undistort .
docker run --rm -it \
    undistort /code/undistort.py --scale=1.5 /code/examples/example.png
```

The command above writes the output files to the container's internal
filesystem, which is not easily accessible from the host. One way to
work around this is to mount the host `examples` folder into the
container.
```
docker run --rm -it                       \
    -v $PWD/examples/:/code/examples/     \
    undistort /code/undistort.py --scale=1.5 /code/examples/example.png
```

This command runs the undistortion script on the `examples/example.png`
and produces the undistortion table `examples/example.png.table` and
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
