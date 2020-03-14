FROM debian:buster-slim AS builder

ENV MPICH_VER=3.3.2
ENV MPICH_NAME=mpich-3.3.2
ENV MPICH_DIR=/usr/local/mpich-3.3.2

ENV HDF5_BASE=hdf5-1.12
ENV HDF5_NAME=hdf5-1.12.0
ENV HDF5_DIR=/usr/local/hdf5-1.12.0

ENV HYPRE_NAME=hypre-2.11.2
ENV HYPRE_DIR=/usr/local/hypre-2.11.2

RUN apt-get update &&\
    apt-get install -yq gcc build-essential wget tar &&\
    apt-get clean &&\
    wget https://www.mpich.org/static/downloads/${MPICH_VER}/${MPICH_NAME}.tar.gz &&\
    tar -xvzf ${MPICH_NAME}.tar.gz &&\
    cd ${MPICH_NAME} &&\
    ./configure --prefix=${MPICH_DIR} CFLGS="-D_LAGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64" --disable-fortran &&\
    make &&\
#    make check &&\
    make install &&\
    export PATH="${MPICH_DIR}/bin:$PATH" &&\
    cd .. &&\
    wget https://support.hdfgroup.org/ftp/HDF5/releases/${HDF5_BASE}/${HDF5_NAME}/src/${HDF5_NAME}.tar.gz &&\
    tar -xvzf ${HDF5_NAME}.tar.gz &&\
    cd ${HDF5_NAME} &&\
    CC=mpicc ./configure --prefix=${HDF5_DIR} --enable-parallel &&\
    make &&\
#    make check &&\
    make install &&\
		export PATH="${HDF5_DIR}/bin:$PATH" &&\
		cd .. &&\
		wget https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/download/${HYPRE_NAME}.tar.gz &&\
		tar -xvzf ${HYPRE_NAME}.tar.gz &&\
		cd ${HYPRE_NAME}/src &&\
		./configure --prefix=${HYPRE_DIR} &&\
		make install &&\
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#-------------------------------------------------------------------------
FROM debian:buster-slim

ENV MPICH_DIR=/usr/local/mpich-3.3.2
ENV HDF5_DIR=/usr/local/hdf5-1.12.0
ENV HYPRE_DIR=/usr/local/hypre-2.11.2

ENV GSL_NAME=gsl-2.6

ENV PATH="${MPICH_DIR}/bin:${HDF5_DIR}/bin:${PATH}"

ENV LD_LIBRARY_PATH=/usr/local/lib

RUN apt-get update &&\
		apt-get install -yq build-essential wget tar less &&\
		wget ftp://ftp.gnu.org/gnu/gsl/${GSL_NAME}.tar.gz &&\
		tar -xvzf ${GSL_NAME}.tar.gz &&\
		cd ${GSL_NAME} &&\
		./configure &&\
		make &&\
		make install &&\
		rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder ${MPICH_DIR} ${MPICH_DIR}
COPY --from=builder ${HDF5_DIR} ${HDF5_DIR}
COPY --from=builder ${HYPRE_DIR} ${HYPRE_DIR}

ENV DIRPATH=/HYPRE-GRF

WORKDIR ${DIRPATH}

COPY src src
COPY include include
COPY Makefile Makefile