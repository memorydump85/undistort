undistort
=========

Computes an undistortion table using locally-weighted homographies.
The method used is described in:

> [_Locally-weighted Homographies for Calibration of Imaging Systems_](http://april.eecs.umich.edu/papers/details.php?name=ranganathan2014iros)<br/>
> Pradeep Ranganathan and Edwin Olson


Required python packages
------------------------
```
numpy >= 1.9.0
scipy >= 0.16.1
scikit-learn >= 0.16.1
scikit-image >= 0.11.3
cython >= 0.22.1
```


How To
-------

Install the required dependencies:
```bash
pip install numpy scipy scikit-learn scikit-image cython
```

Clone and build:
```bash
git clone --recursive git@github.com:memorydump85/undistort.git
cd undistort/apriltag
make
cd ..
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

Running the code using Docker
----
Using the following `undistort.Dockerfile`:
```docker
FROM ubuntu:16.04

RUN \
  apt-get update && \
  apt -y install build-essential  pkg-config git python python-pip python-numpy && \
  python -m pip -U pip && \
  pip install -U numpy scipy scikit-learn scikit-image cython && \
  cd /root && \
  git clone --recursive https://github.com/memorydump85/undistort.git && \
  cd /root/undistort/apriltag && \
  make -j
```
build a docker image and run an interactive container using that image:
```bash
docker build -t undistort -f undistort.Dockerfile
docker run -it undistort
```
Then, run the `undistort.py` script inside the container:
```bash
cd /root/undistort
./undistort.py --scale=1.5 examples/example.png
```

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

