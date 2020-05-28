FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y \
    apt-utils \
    sudo \
    build-essential \
    gcc \
    g++ \
    gfortran \
    gdb \
    wget \
    curl \
    tar \
    git \
    vim \
    make \
    cmake \
    cmake-curses-gui \
    python3-pip \
    python3-dev \
    libssl-dev \
    libnetcdf-dev \
    libnetcdff-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives  --set python /usr/bin/python3 && \
    update-alternatives  --set pip /usr/bin/pip3

#### START INSTALL MPICH #######################################
# Source is available at http://www.mpich.org/static/downloads/
# Build Options:
# See installation guide of target MPICH version
# Ex: http://www.mpich.org/static/downloads/3.2/mpich-3.2-installguide.pdf
# These options are passed to the steps below
ARG MPICH_VERSION="3.2"
ARG MPICH_CONFIGURE_OPTIONS=""
ARG MPICH_MAKE_OPTIONS=""

# Download, build, and install MPICH
RUN mkdir /tmp/mpich-src
WORKDIR /tmp/mpich-src
RUN wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz \
      && tar xfz mpich-${MPICH_VERSION}.tar.gz  \
      && cd mpich-${MPICH_VERSION}  \
      && ./configure ${MPICH_CONFIGURE_OPTIONS}  \
      && make ${MPICH_MAKE_OPTIONS} && make install \
      && rm -rf /tmp/mpich-src

#### CLEAN UP ####
WORKDIR /
RUN rm -rf /tmp/*
#### END INSTALL MPICH #######################################

RUN pip install --upgrade pip \
  && pip install \
    pytest \
    numpy \
    matplotlib \
    mpi4py

# set TZ
ENV TZ=US/Pacific
RUN echo $TZ > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

# add default user
ARG USER=mpi
ENV USER ${USER}
RUN useradd -ms /bin/bash ${USER} \
      && echo "${USER}   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
ENV USER_HOME /home/${USER}
RUN chown -R ${USER}:${USER} ${USER_HOME}

# create working directory
ARG WORKDIR=/work
ENV WORKDIR ${WORKDIR}
RUN mkdir ${WORKDIR}
RUN chown -R ${USER}:${USER} ${WORKDIR}

WORKDIR ${WORKDIR}
USER ${USER}

CMD ["/bin/bash"]

