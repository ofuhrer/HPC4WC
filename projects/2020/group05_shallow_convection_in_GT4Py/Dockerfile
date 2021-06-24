FROM ubuntu:20.04

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
    libboost-all-dev \
    libnetcdf-dev \
    libnetcdff-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives  --set python /usr/bin/python3 && \
    update-alternatives  --set pip /usr/bin/pip3
	
# set TZ
ENV TZ=US/Pacific
RUN echo $TZ > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

# install serialbox from source
RUN git clone --single-branch --branch savepoint_as_string https://github.com/VulcanClimateModeling/serialbox2.git /serialbox
RUN cd /serialbox && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/serialbox -DCMAKE_BUILD_TYPE=Release \
          -DSERIALBOX_USE_NETCDF=ON -DSERIALBOX_ENABLE_FORTRAN=ON \
          -DSERIALBOX_TESTING=ON  ../ && \
    make -j8 && \
    make test && \
    make install && \
    /bin/rm -rf /serialbox
	
# gt4py
RUN pip install git+https://github.com/gridtools/gt4py.git \
    && python -m gt4py.gt_src_manager install

# add default user
ARG USER=user
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

ENV PYTHONPATH="/usr/local/serialbox/python:${PYTHONPATH}"

CMD ["/bin/bash"]

