/*
  Compile with:   make grf
  
  Sample run:     mpirun -np 4 grf -n 32 -solver 0 -v 1 1
                  mpiexec -n 4 ./grf -n 32 -solver 0 -v 1 1
  
  To see options: grf -help
*/

#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"
#include "hdf5_utils.h"

double ksq(double x0, double x1, double x2);

double gam(double x0, double x1, double x2);

double bet1(double x0, double x1, double x2);

double bet2(double x0, double x1, double x2);

void coeff_values(double* coeff, double x0, double x1, double x2,
		  double dx, double dy, double dz, int index);

int main (int argc, char *argv[])
{
  int i, j, k;
  
  int myid, num_procs;
  
  int ni, nj, nk, pi, pj, pk, npi, npj, npk;
  double dx, dy, dz;
  //  double mass, mdot, rmin, rmax, period;
  int ilower[3], iupper[3];
  
  int solver_id;
  int n_pre, n_post;
  
  clock_t start_t = clock();
  clock_t check_t;
  
  HYPRE_StructGrid     grid;
  HYPRE_StructStencil  stencil;
  HYPRE_StructMatrix   A;
  HYPRE_StructVector   b;
  HYPRE_StructVector   x;
  HYPRE_StructSolver   solver;
  HYPRE_StructSolver   precond;
  
  int num_iterations;
  double final_res_norm;
  
  int output, timer;
  
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  
  /* Set defaults */
  ni  = 32; /* nj and nk are set equal to ni later */
  npi = 1;
  npj = 1;
  npk = num_procs; /* default processor grid is 1 x 1 x N */
  solver_id = 0;
  n_pre  = 1;
  n_post = 1;
  output = 1; /* output data by default */
  timer  = 0;

  // TODO read parameters as inputs
  /* mass   = 1.E6; /\* in solar masses*\/ */
  /* mdot   = 1.E-7; /\* in solar masses per year *\/ */
  /* rmin   = 3.; /\* radius and time in terms of M *\/ */
  /* rmax   = 10.;  */
  /* period = 20000. */
  
  /* Initiialize rng */
  const gsl_rng_type *T;
  gsl_rng *rstate;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  rstate = gsl_rng_alloc(T);
  gsl_rng_set(rstate, gsl_rng_default_seed + myid);	 
  
  /* Parse command line */
  {
    int arg_index   = 0;
    int print_usage = 0;
    int check_pgrid = 0;
    
    while (arg_index < argc) {
      if ( strcmp(argv[arg_index], "-n") == 0 ) {
	arg_index++;
	ni = atoi(argv[arg_index++]);
      }
      else {
	arg_index++;
      }
    }
    nj = ni;
    nk = ni;
    
    arg_index = 0;
    while (arg_index < argc) {
      if ( strcmp(argv[arg_index], "-ni") == 0 ) {
	arg_index++;
	ni = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nj") == 0 ) {
	arg_index++;
	nj = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nk") == 0 ) {
	arg_index++;
	nk = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pgrid") == 0 ) {
	arg_index++;
	if (arg_index >= argc - 2) {
	  check_pgrid = 1;
	  break;
	}
	npi = atoi(argv[arg_index++]);
	npj = atoi(argv[arg_index++]);
	npk = atoi(argv[arg_index++]);
	if ( num_procs != (npi * npj * npk) ) {
	  check_pgrid = 1;
	  break;
	}
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 ) {
	arg_index++;
	solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-v") == 0 ) {
	arg_index++;
	n_pre = atoi(argv[arg_index++]);
	n_post = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dryrun") == 0 ) {
	arg_index++;
	output = 0;
      }
      else if ( strcmp(argv[arg_index], "-timer") == 0 ) {
	arg_index++;
	timer = 1;
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 ) {
	print_usage = 1;
	break;
      }
      else {
	arg_index++;
      }
    }
    
    if (print_usage) {
      if (myid == 0) {
	printf("\n");
	printf("Usage: %s [<options>]\n", argv[0]);
	printf("\n");
	printf("  -n <n>                : General grid side length per processor (default: 32).\n");
	printf("  -ni <n> (or nj/nk)    : Grid side length for specific side per processor\n");
	printf("                          (default: 32). If both -n and -ni are specified, ni\n");
	printf("                          will overwrite n for that side only. \n");
	printf("  -pgrid <pi> <pj> <pk> : Layout of processor grid. (default: 1 x 1 x num_procs)\n");
	printf("                          The product of pi * pj * pk must equal num_proc.\n");
	printf("  -solver <ID>          : solver ID\n");
	printf("                          0  - PCG with SMG precond (default)\n");
	printf("                          1  - SMG\n");
	printf("  -v <n_pre> <n_post>   : Number of pre and post relaxations (default: 1 1).\n");
	printf("  -dryrun               : Run solver w/o data output.\n");
	printf("  -timer                : Time each step on processor zero.\n");
	printf("\n");
      }
      MPI_Finalize();
      return (0);
    }
    
    if (check_pgrid) {
      if (myid ==0) {
	printf("Error: Processor grid does not match the total number of processors. \n");
	printf("       npi * npj * npk must equal num_proc (see -help). \n");
      }
      MPI_Finalize();
      return (0);
    }	
  }
  
  /* Figure out the processor grid (npi x npj x npk).  The local problem
     size for the interior nodes is indicated by (ni x nj x nk).
     pi and pj and pk indicate position in the processor grid. */
  dx = 2. * M_PI / (npi * ni); 
  dy = 2. * M_PI / (npj * nj);
  dz = 2. * M_PI / (128);
  
  pk = myid / (npi * npj);
  pj = (myid - pk * npi * npj) / npi;
  pi = myid - pk * npi * npj - pj * npi;
  
  /* Figure out the extents of each processor's piece of the grid. */
  ilower[0] = pi * ni;
  ilower[1] = pj * nj;
  ilower[2] = pk * nk;
  
  iupper[0] = ilower[0] + ni - 1;
  iupper[1] = ilower[1] + nj - 1;
  iupper[2] = ilower[2] + nk - 1;
  
  /* 1. Set up a grid */
  {
    /* Create an empty 3D grid object */
    HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid);
    
    /* Add a new box to the grid */
    HYPRE_StructGridSetExtents(grid, ilower, iupper);
    
    /* Set periodic boundary conditions on t and phi*/
    int boundcon[3] = {0, npj * nj, npk * nk};
    HYPRE_StructGridSetPeriodic(grid, boundcon);
    
    /* This is a collective call finalizing the grid assembly.
       The grid is now ``ready to be used'' */
    HYPRE_StructGridAssemble(grid);
  }
  
  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Stencils initialized: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);
  
#define NSTENCIL 19
  /* 2. Define the discretization stencil */
  {
    /* Create an empty 3D, 19-pt stencil object */
    HYPRE_StructStencilCreate(3, NSTENCIL, &stencil);
    
    /* Define the geometry of the stencil */
    /*
      22 14 21    10 04 09    26 18 25    ^
      11 05 12    01 00 02    15 06 16    |
      19 13 20    07 03 08    23 17 24    j i ->    k - - >
      
      Delete zero entries:
      xx 14 xx    10 04 09    xx 18 xx    ^
      11 05 12    01 00 02    15 06 16    |			 
      xx 13 xx    07 03 08    xx 17 xx    j i ->    k - - >			
    */
    {
      int entry;
      int offsets[NSTENCIL][3] = {{0,0,0},
				  {-1,0,0}, {1,0,0}, 
				  {0,-1,0}, {0,1,0}, 
				  {0,0,-1}, {0,0,1}, 
				  {-1,-1,0}, {1,-1,0}, 
				  {1,1,0}, {-1,1,0}, 
				  {-1,0,-1}, {1,0,-1}, 
				  {0,-1,-1}, {0,1,-1}, 
				  {-1,0,1}, {1,0,1}, 
				  {0,-1,1}, {0,1,1}
				  /* {-1,-1,-1}, {1,-1,-1}, */
				  /* {1,1,-1}, {-1,1,-1}, */
				  /* {-1,-1,1}, {1,-1,1}, */
				  /* {1,1,1}, {-1,1,1} */
      };
			
      for (entry = 0; entry < NSTENCIL; entry++)
	HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
    }
  }

  /* 3. Set up a Struct Matrix */
  {
    int nentries = NSTENCIL;
    int nvalues = nentries * ni * nj * nk;
    double *values;
    int stencil_indices[NSTENCIL];
		
    /* Create an empty matrix object */
    HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);
		
    /* Indicate that the matrix coefficients are ready to be set */
    HYPRE_StructMatrixInitialize(A);

    values = (double*) calloc(nvalues, sizeof(double));

    for (j = 0; j < nentries; j++)
      stencil_indices[j] = j;

    /* Set the standard stencil at each grid point,
       we will fix the boundaries later */
    for (i = 0; i < nvalues; i += nentries) {
      double x0, x1, x2;
      double coeff[6];
      int gridi, gridj, gridk, temp;

      temp = i / nentries;
      gridk = temp / (ni * nj);
      gridj = (temp - ni * nj * gridk) / ni;
      gridi = temp - ni * nj * gridk + (pi - gridj) * ni;
      gridj += pj * nj;
      gridk += pk * nk;
			
      x0 = dx * gridi;
      x1 = dy * gridj;
      x2 = dz * gridk;
			
      coeff_values(coeff, x0, x1, x2, dx, dy, dz, 6);
		  			
      /*0=a, 1=b, 2=c, 3=d, 4=e, 5=f*/
      /*
	xx 14 xx    10 04 09    xx 18 xx    ^
	11 05 12    01 00 02    15 06 16    |			 
	xx 13 xx    07 03 08    xx 17 xx    j i ->    k - - >			
      */
			
      values[i]    = coeff[5];
      values[i+1]  = coeff[0];
      values[i+2]  = coeff[0];
      values[i+3]  = coeff[2];
      values[i+4]  = coeff[2];
      values[i+5]  = coeff[4];
      values[i+6]  = coeff[4];
      values[i+7]  = coeff[1];
      values[i+8]  = -coeff[1];
      values[i+9]  = coeff[1];
      values[i+10] = -coeff[1];
      values[i+11] = 0.;
      values[i+12] = 0.;
      values[i+13] = coeff[3];
      values[i+14] = -coeff[3];
      values[i+15] = 0.;
      values[i+16] = 0.;
      values[i+17] = -coeff[3];
      values[i+18] = coeff[3];
			
      /* values[i+19] = 0.; */
      /* values[i+20] = 0.; */
      /* values[i+21] = 0.; */
      /* values[i+22] = 0.; */
      /* values[i+23] = 0.; */
      /* values[i+24] = 0.; */
      /* values[i+25] = 0.; */
      /* values[i+26] = 0.; */
    }

    HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
				   stencil_indices, values);
		
    free(values);
  }

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Stencils values set: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);
	
  /* 4. Incorporate the boundary conditions: go along each edge of
     the domain and set the stencil entry that reaches to the boundary.*/
  {
    int bc_ilower[3];
    int bc_iupper[3];
    int	nentries = 6;
    int nvalues  = nentries * nj * nk; /* number of stencil entries times the 
					  length of one side of my grid box */
    double *values;
    int stencil_indices[nentries];
    values = (double*) calloc(nvalues, sizeof(double));
		
    /* Recall: pi and pj describe position in the processor grid */
    if (pi == 0) {
      /* Bottom row of grid points */
      double coeff[6];

      coeff_values(coeff, 0., 0., 0., dx, dy, dz, 6);
			
      for (j = 0; j < nvalues; j += nentries) {
	values[j]   = coeff[5] + coeff[0];
	values[j+1] = 0.0;
	values[j+2] = coeff[2] + coeff[1];
	values[j+3] = coeff[2] - coeff[1];
	values[j+4] = 0.0;
	values[j+5] = 0.0;
      }
			
      bc_ilower[0] = pi * ni;
      bc_ilower[1] = pj * nj;
      bc_ilower[2] = pk * nk;
			
      bc_iupper[0] = bc_ilower[0];
      bc_iupper[1] = bc_ilower[1] + nj - 1;
      bc_iupper[2] = bc_ilower[2] + nk - 1;
			
      stencil_indices[0] = 0;
      stencil_indices[1] = 1;
      stencil_indices[2] = 3;
      stencil_indices[3] = 4;
      stencil_indices[4] = 7;
      stencil_indices[5] = 10;
			
      HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
				     stencil_indices, values);
    }
		
    if (pi == npi - 1) {
      /* upper row of grid points */
      double coeff[6];
			
      bc_ilower[0] = pi * ni + ni - 1;
      bc_ilower[1] = pj * nj;
      bc_ilower[2] = pk * nk;
			
      bc_iupper[0] = bc_ilower[0];
      bc_iupper[1] = bc_ilower[1] + nj - 1;
      bc_iupper[2] = bc_ilower[2] + nk - 1;

      coeff_values(coeff, bc_ilower[0] * dx, 0., 0., dx, dy, dz, 6);
		  
      for (j = 0; j < nvalues; j += nentries) {
	values[j]   = coeff[5] + coeff[0];
	values[j+1] = 0.0;
	values[j+2] = coeff[2] - coeff[1];
	values[j+3] = coeff[2] + coeff[1];
	values[j+4] = 0.0;
	values[j+5] = 0.0;
      }
			
      stencil_indices[0] = 0;
      stencil_indices[1] = 2;
      stencil_indices[2] = 3;
      stencil_indices[3] = 4;
      stencil_indices[4] = 8;
      stencil_indices[5] = 9;
			
      HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
				     stencil_indices, values);
    }
		
    free(values);
  }
	
  /* This is a collective call finalizing the matrix assembly.
     The matrix is now ``ready to be used'' */
  HYPRE_StructMatrixAssemble(A);

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Boundary conditions set: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);
	
  /* 5. Set up Struct Vectors for b and x */
  {
    int    nvalues = ni * nj * nk;
    double *values;
		
    values = (double*) calloc(nvalues, sizeof(double));
		
    /* Create an empty vector object */
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);
		
    /* Indicate that the vector coefficients are ready to be set */
    HYPRE_StructVectorInitialize(b);
    HYPRE_StructVectorInitialize(x);
		
    /* Set the values */
    for (i = 0; i < nvalues; i++) {
      values[i] = gsl_ran_gaussian(rstate, 1.);
    }
    HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);
		
    for (i = 0; i < nvalues; i ++)
      values[i] = 0.0;
    HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
		
    free(values);
		
    /* This is a collective call finalizing the vector assembly.
       The vector is now ``ready to be used'' */
    HYPRE_StructVectorAssemble(b);
    HYPRE_StructVectorAssemble(x);
  }

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Struct vector assembled: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);
  
  /* 6. Set up and use a struct solver
     (Solver options can be found in the Reference Manual.) */
  if (solver_id == 0) {
    HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructPCGSetMaxIter(solver, 50 );
    HYPRE_StructPCGSetTol(solver, 1.0e-06 );
    HYPRE_StructPCGSetTwoNorm(solver, 1 );
    HYPRE_StructPCGSetRelChange(solver, 0 );
    HYPRE_StructPCGSetPrintLevel(solver, 2 ); /* print each CG iteration */
    HYPRE_StructPCGSetLogging(solver, 1);
		
    /* Use symmetric SMG as preconditioner */
    HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
    HYPRE_StructSMGSetMemoryUse(precond, 0);
    HYPRE_StructSMGSetMaxIter(precond, 1);
    HYPRE_StructSMGSetTol(precond, 0.0);
    HYPRE_StructSMGSetZeroGuess(precond);
    HYPRE_StructSMGSetNumPreRelax(precond, 1);
    HYPRE_StructSMGSetNumPostRelax(precond, 1);
		
    /* Set the preconditioner and solve */
    HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve,
			      HYPRE_StructSMGSetup, precond);
    HYPRE_StructPCGSetup(solver, A, b, x);
    HYPRE_StructPCGSolve(solver, A, b, x);

    /* Get some info on the run */
    HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
    HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
		
    /* Clean up */
    HYPRE_StructPCGDestroy(solver);
  }
	
  if (solver_id == 1) {
    HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructSMGSetMemoryUse(solver, 0);
    HYPRE_StructSMGSetMaxIter(solver, 50);
    HYPRE_StructSMGSetTol(solver, 1.0e-06);
    HYPRE_StructSMGSetRelChange(solver, 0);
    HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
    HYPRE_StructSMGSetNumPostRelax(solver, n_post);
    /* Logging must be on to get iterations and residual norm info below */
    HYPRE_StructSMGSetPrintLevel(solver, 2);
    HYPRE_StructSMGSetLogging(solver, 1);

    /* Setup and solve */
    HYPRE_StructSMGSetup(solver, A, b, x);
    HYPRE_StructSMGSolve(solver, A, b, x);
		
    /* Get some info on the run */
    HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
    HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
		
    /* Clean up */
    HYPRE_StructSMGDestroy(solver);
  }

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Solver finished: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);	
  
  if (myid == 0) {
    printf("\n");
    printf("Iterations = %d\n", num_iterations);
    printf("Final Relative Residual Norm = %g\n", final_res_norm);
    printf("\n");
  }	

  /* Output data */
  if (output) {
    /* get the local solution */
    int nvalues    = ni * nj * nk;
    double *values = (double*)calloc(nvalues, sizeof(double));
    
    HYPRE_StructVectorGetBoxValues(x, ilower, iupper, values);
    
    /* find the min, max, and ligthcurve */
    double localmin = values[0];
    double localmax = values[0];
    double globalmin, globalmax;
    
    double *local_lc  = (double*)calloc(npk * nk, sizeof(double));
    double *global_lc = (double*)calloc(npk * nk, sizeof(double));
    
    i = 0;
    for (k = pk * nk; k < (pk + 1) * nk; k++) {
      for (j = 0; j < ni * nj; j++) {
	local_lc[k] += values[i];
	if (values[i] < localmin)
	  localmin = values[i];
	if (values[i] > localmax)
	  localmax = values[i];
	i++;
      }
    }
    
    MPI_Allreduce(&localmin, &globalmin, 1, MPI_DOUBLE,
    	       MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localmax, &globalmax, 1, MPI_DOUBLE,
    	       MPI_MAX, MPI_COMM_WORLD);
    MPI_Reduce(local_lc, global_lc, npk * nk, MPI_DOUBLE,
    	       MPI_SUM, 0, MPI_COMM_WORLD);
    
    char filename[255];		
    
    if (myid == 0) {
      time_t rawtime;
      struct tm * timeinfo;
      char buffer[255];
      
      time(&rawtime);
      timeinfo = localtime(&rawtime);
      strftime(buffer, 255, "%Y_%m_%d_%H%M%S", timeinfo);
      
      sprintf(filename, "%s/%d_%d_%d_%s.h5", "output",
	      npi * ni, npj * nj, npk * nk, buffer);
    }
    
    MPI_Bcast(&filename, 255, MPI_CHAR, 0, MPI_COMM_WORLD);
    hdf5_create(filename);
    
    /* save solution to output file*/
    hdf5_set_directory("/");
    hdf5_make_directory("data");
    hdf5_set_directory("/data/");

    /* note: HYPRE has k as the slowest varying, opposite of HDF5 */
    {
      hsize_t fdims[3]  = {npk * nk, npj * nj, npi * ni};
      hsize_t fstart[3] = {pk * nk, pj * nj, pi * ni};
      hsize_t fcount[3] = {nk, nj, ni};
      hsize_t mdims[3]  = {nk, nj, ni};
      hsize_t mstart[3] = {0, 0, 0};
      hdf5_write_array(values, "data", 3, fdims, fstart, fcount,
		       mdims, mstart, H5T_NATIVE_DOUBLE);
    }

    /* output lightcurve and parameters */
    {
      hsize_t fdims  = npk * nk;
      hsize_t fstart = 0;
      hsize_t fcount = 0;
      hsize_t mdims  = 0;
      hsize_t mstart = 0;

      if (myid == 0) {
	fcount = npk * nk;
	mdims  = npk * nk;
      }
	
      hdf5_write_array(global_lc, "lightcurve", 1, &fdims, &fstart, &fcount,
		       &mdims, &mstart, H5T_NATIVE_DOUBLE);
    }
    
    hdf5_set_directory("/");
    hdf5_make_directory("params");
    hdf5_set_directory("/params/");
    
    hdf5_write_single_val(&npi, "npi", H5T_STD_I32LE);
    hdf5_write_single_val(&npj, "npj", H5T_STD_I32LE);
    hdf5_write_single_val(&npk, "npk", H5T_STD_I32LE);
    hdf5_write_single_val(&ni, "ni", H5T_STD_I32LE);
    hdf5_write_single_val(&nj, "nj", H5T_STD_I32LE);
    hdf5_write_single_val(&nk, "nk", H5T_STD_I32LE);
    hdf5_write_single_val(&gsl_rng_default_seed, "seed", H5T_STD_U64LE);
    
    hdf5_write_single_val(&globalmin, "min", H5T_IEEE_F64LE);
    hdf5_write_single_val(&globalmax, "max", H5T_IEEE_F64LE);
    
    hdf5_close();
    
    free(values);
    free(local_lc);
    free(global_lc);
  }
  
  check_t = clock();
  if ( (myid == 0) && (output) && (timer) )
    printf("Data output: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);
	
  /* Free memory */
  HYPRE_StructGridDestroy(grid);
  HYPRE_StructStencilDestroy(stencil);
  HYPRE_StructMatrixDestroy(A);
  HYPRE_StructVectorDestroy(b);
  HYPRE_StructVectorDestroy(x);
	
  /* Finalize MPI */
  MPI_Finalize();
	
  gsl_rng_free(rstate);
	
  return (0);
}

double ksq(double x0, double x1, double x2)
{
  return 1.;// + log(1. + x0 * log(10.) / (2. * M_PI) );
}

double gam(double x0, double x1, double x2)
{
  return 1.;// * (1. + x0 * log(10.) / (2. * M_PI));
}

double bet1(double x0, double x1, double x2)
{
  return 36.;// * (1. - 0.75 * erf(0.5 * x0 * log(10.) / (2 * M_PI) ));
}

double bet2(double x0, double x1, double x2)
{
  return 100.;
}

void coeff_values(double* coeff, double x0, double x1, double x2, double dx, double dy, double dz, int index)
{
  double theta, psi, gamma, beta1, beta2;
  theta = -7. * M_PI / 18.;
  psi   = atan( exp( 4. - 1.5 * ( 2. * dx + x0 / 2. ) ) );
  gamma = gam(x0, x1, x2);
  beta1 = bet1(x0, x1, x2);
  beta2 = bet2(x0, x1, x2);
  coeff[0] = ( gamma + beta1 * cos(theta) * cos(theta) ) / (dx * dx);
  coeff[1] = 0.5 * beta1 * cos(theta) * sin(theta) / (dx * dy);
  coeff[2] = ( gamma + beta1 * sin(theta) * sin(theta)
	       + beta2 * sin(psi) * sin(psi) ) / (dy * dy);
  coeff[3] = 0.5 * beta2 * cos(psi) * sin(psi) / (dy * dz);
  coeff[4] = ( gamma + beta2 * cos(psi) * cos(psi) ) / (dz * dz);
  coeff[5] = -2. * ( coeff[0] + coeff[2] + coeff[4] ) - ksq(x0, x1, x2);
}
