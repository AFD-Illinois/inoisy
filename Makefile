# Compiler and hypre location
CC        = h5pcc
ifndef HYPRE_DIR
$(info HYPRE_DIR not defined, trying '/usr/local/hypre')
HYPRE_DIR = /usr/local/hypre
endif

# Local directories
INC_DIR = $(CURDIR)/include
SRC_DIR = $(CURDIR)/src
OBJ_DIR = $(CURDIR)/obj

# Compiling and linking options
COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include -I$(INC_DIR)
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)

LINKOPTS = $(COPTS)
LDFLAGS  = -L$(HYPRE_DIR)/lib
LIBS     = -lHYPRE -lm -lgsl -lgslcblas -shlib -lstdc++
LFLAGS   = $(LINKOPTS) $(LIBS)

# List of all programs to be compiled

EXE = poisson disk_logr disk_xy noisy_unif noisy_disk general_xy

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
