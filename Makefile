########################################################################
# Compiler and external dependences
########################################################################
CC        = mpicc
HYPRE_DIR = /usr/local/hypre

########################################################################
# Compiling and linking options
########################################################################
COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)
# FOPTS     = -g
# FINCLUDES = $(CINCLUDES)
# FFLAGS    = $(FOPTS) $(FINCLUDES)
# CXXOPTS   = $(COPTS) -Wno-deprecated
# CXXINCLUDES = $(CINCLUDES) -I..
# CXXDEFS   = $(CDEFS)
# IFLAGS_BXX = 
# CXXFLAGS  = $(CXXOPTS) $(CXXINCLUDES) $(CXXDEFS) $(IFLAGS_BXX)
# IF90FLAGS = 
# F90FLAGS = $(FFLAGS) $(IF90FLAGS)


LINKOPTS  = $(COPTS)
LIBS      = -L$(HYPRE_DIR)/lib -lHYPRE -lm -lgsl -lgslcblas
LFLAGS    = $(LINKOPTS) $(LIBS) -lstdc++
# LFLAGS_B =\
#  -L${HYPRE_DIR}/lib\
#  -lbHYPREClient-C\
#  -lbHYPREClient-CX\
#  -lbHYPREClient-F\
#  -lbHYPRE\
#  -lsidl -ldl -lxml2

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
grf: grf.o
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
	cd vis; make clean
distclean: clean
	rm -f $(ALLPROGS) $(ALLPROGS:=*~)
	rm -fr README*
