# HYPRE-GRF
2+1D GRFs of accretion disks generated by HYPRE (version 2.11.2)

Generates Gaussian random fields (GRFs) following Lindgren et al. 2011.

Requires HYPRE, HDF5, and MPI.

Compile with:   make all
                or make <model name> e.g. make disk
  
Sample run:     mpirun -np 8 disk -n 64 -solver 1 -pgrid 4 2 1 -output data
                mpiexec -n 4 ./poisson -n 32 -solver 0 -timer -dryrun

To see options: use option help, -help, or --help

The shape of the processor grid and the size of the grid assigned to each
processor is decided at runtime with the options -pgrid, -n, -ni, -nj, and
-nk. If both -n and -ni (or nj or nk) are specified, -ni will overwrite -n
for that specific side.

Other parameters such as domain size, correlation lengths, envelope
functions, and source terms are determined by 'param_<model_name>.c'. Edit
the file and run make again for different parmeters.

Boundary conditions, stencil structure, etc. are in 'model_<model_name>.c'.

The output is in
'<output_directory>/<model_name>_<grid_dimensions>_<date_and_time>.h5'
<output_directory> is specified by the option -output (or -o) (default '.')

To add a model, create a 'param_<model_name>.c' and 'model_<model_name>.c'
following the structure of the other models, and add the <model_name> to the
list of models in the Makefile.

Current models: poisson, disk, noisy_uniform, noisy_disk

The default parameters for poisson are useful for testing the program. The 
solution should match sin(x0)sin(x1)sin(x2) on a periodic domain {0,2Pi} x
{0,2Pi} x {0,2Pi}. 

Index naming conventions:
Given a stencil {i,j,k}, HYPRE has k as the slowest varying index. 
Thus, the indices correspond to x0 = k, x1 = j, x2 = i. 

13 Feb 2020
