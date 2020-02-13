# Compiler and external dependences
CC        = h5pcc
HYPRE_DIR = /home/dl6/hypre-2.11.2/src/hypre

# Local directories
INC_DIR = $(CURDIR)/include
SRC_DIR = $(CURDIR)/src
OBJ_DIR = $(CURDIR)/obj

# Compiling and linking options
COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include -I/home/dl6/HYPRE-GRF -I$(INC_DIR)
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)

LINKOPTS = $(COPTS)
LDFLAGS  = -L$(HYPRE_DIR)/lib
LIBS     = -lHYPRE -lm -lgsl -lgslcblas -shlib -lstdc++
LFLAGS   = $(LINKOPTS) $(LIBS)

# List of all programs to be compiled

EXE = disk poisson

SRC := $(addprefix $(SRC_DIR)/,main.c hdf5_utils.c model_%.c param_%.c)
OBJ := $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(EXE)

$(EXE): %: $(OBJ)
	$(CC) $(LDFLAGS) $^ $(LFLAGS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir $@

default: all

# Clean up

clean:
	$(RM) -r $(OBJ_DIR)
distclean: clean
	$(RM) $(EXE)
