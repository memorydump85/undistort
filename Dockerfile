FROM ubuntu:19.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt -y install \
        build-essential pkg-config \
        python3-dev python3-pip \
        python3-numpy python3-scipy cython3 python3-sklearn python3-skimage

ADD ./ /code/
RUN make -j -C /code/apriltag clean all
WORKDIR /code