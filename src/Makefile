CC=gcc
LD=gcc
EXEC=axbc2d

#CUDA_PATH use CUDATOOLKIT_HOME for CRAY
CUDA_HOME=/usr/local/cuda
MPI_HOME=/usr/lib/openmpi




CUDACC=$(CUDA_HOME)/bin/nvcc 
CUDA_ARCH=-arch=sm_37
CUDACFLAGS=-m64 -c -O3  --ptxas-options=-v -DCUDA  -DONEDEGREE --use_fast_math


CFLAGS=-g -W -Wall -Wno-unused-function -Wno-unused-parameter -c -O3 -DMPI -DCUDA  -L$(MPI_HOME)/lib64 -lmpi   -I$(CUDA_HOME)/include
LDFLAGS = -L$(MPI_HOME)/lib64 -lmpi -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ ../generator/libgraph_generator_mpi.a  -lm -Wno-deprecated-gpu-targets

OBJ=axbc2d.o cuda_kernels.o adtp.o

# https://stackoverflow.com/questions/2734719/how-to-compile-a-static-library-in-linux
axbc: ${OBJ}
	${LD} -o ../bin/$(EXEC) ${OBJ} ${LDFLAGS}

.c.o:; $(CC) $(CFLAGS) -I$(MPI_HOME)/include    $< -o $@

%.o: %.cu
	$(CUDACC) $(CUDACFLAGS) $(CUDA_ARCH) -I$(MPI_HOME)/include $<
clean:
	rm -rf *.o ../bin/$(EXEC)

