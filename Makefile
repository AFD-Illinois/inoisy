########################################################################
# Compiler and external dependences
########################################################################
CC        = h5pcc
HYPRE_DIR = /home/dl6/hypre-2.11.2/src/hypre

########################################################################
# Compiling and linking options
########################################################################
COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)



LINKOPTS  = $(COPTS)
LIBS      = -L$(HYPRE_DIR)/lib -lHYPRE -lm -lgsl -lgslcblas -shlib
LFLAGS    = $(LINKOPTS) $(LIBS) -lstdc++

########################################################################
# Rules for compiling the source files
########################################################################
.SUFFIXES: .c

.c.o:
	$(CC) $(CFLAGS) -c $<

########################################################################
# List of all programs to be compiled
########################################################################
ALLPROGS = grf poisson

all: $(ALLPROGS)

default: all

########################################################################
# Compile grf
########################################################################
grf: grf.o hdf5_utils.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Compile poisson
########################################################################
poisson: poisson.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Clean up
########################################################################
clean:
	rm -f $(ALLPROGS:=.o)
distclean: clean
	rm -f $(ALLPROGS) $(ALLPROGS:=*~)
	rm -fr README*
