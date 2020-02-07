/*
  Compile with:   make grf
  
  Sample run:     mpirun -np 4 grf -n 32 -solver 0 -v 1 1
                  mpiexec -n 4 ./grf -n 32 -solver 0 -v 1 1
  
  To see options: grf -help

  Index naming conventions:
    x0 = t, x1 = r, x2 = phi
    Given a stencil {i,j,k}, HYPRE has k as the slowest varying index. 
    Thus, the indices correspond to i = x2, j = x1, k = x0. 
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
		  double dx0, double dx1, double dx2, double mass, int index);

int main (int argc, char *argv[])
{
  int i, j, k;
  
  int myid, num_procs;
  
  int ni, nj, nk, pi, pj, pk, npi, npj, npk;
  double dx0, dx1, dx2;
  int ilower[3], iupper[3];
  
  int solver_id;
  int n_pre, n_post;
  
  clock_t start_t = clock();
  clock_t check_t;
  
  HYPRE_StructGrid    grid;
  HYPRE_StructStencil stencil;
  HYPRE_StructMatrix  A;
  HYPRE_StructVector  b;
  HYPRE_StructVector  x;
  HYPRE_StructSolver  solver;
  HYPRE_StructSolver  precond;
  
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

  double mass, mdot, rmin, rmax, period;
  // TODO read parameters as inputs
  mass   = 1.E6; /* in solar masses*/
  mdot   = 1.E-7; /* in solar masses per year */
  rmin   = 2.; /* radius and time in terms of M */
  rmax   = 100.;
  period = 5000.;
 
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
      else if ( strcmp(argv[arg_index], "help")   == 0 ||
		strcmp(argv[arg_index], "-help")  == 0 ||
		strcmp(argv[arg_index], "--help") == 0 ) {
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
	printf("Sample run:     mpirun -np 4 grf -n 32 -nk 64 -pgrid 2 2 1 -solver 0\n");
	printf("                mpiexec -n 4 ./grf -n 32 -nk 64 -pgrid 2 2 1 -solver 0\n");
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
  dx0 = period / (npk * nk);
  dx1 = (log(rmax) - log(rmin)) / (npj * nj); 
  dx2 = 2. * M_PI / (npi * ni);
  
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
    HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid);
    HYPRE_StructGridSetExtents(grid, ilower, iupper);
    
    /* Set periodic boundary conditions on t and phi*/
    int boundcon[3] = {npi * ni, 0, npk * nk};
    HYPRE_StructGridSetPeriodic(grid, boundcon);
    
    HYPRE_StructGridAssemble(grid);
  }
  
  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Stencils initialized: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);
  
#define NSTENCIL 19
  /* 2. Define the discretization stencil */
  {
    HYPRE_StructStencilCreate(3, NSTENCIL, &stencil);
    
    /* Define the geometry of the stencil */
    /* Recall i = x2, j = x1, k = x0 */
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
		
    HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);
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

      x0 = dx0 * gridk;			
      x1 = log(rmin) + dx1 * gridj;
      x2 = dx2 * gridi;
			
      coeff_values(coeff, x0, x1, x2, dx0, dx1, dx2, mass, 6);
		  			
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
      values[i+11] = coeff[3];
      values[i+12] = -coeff[3];
      values[i+13] = 0.;
      values[i+14] = 0.;
      values[i+15] = -coeff[3];
      values[i+16] = coeff[3];
      values[i+17] = 0.;
      values[i+18] = 0.;
			
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
    int nvalues  = nentries * ni * nk; /* number of stencil entries times the 
					  length of one side of my grid box */
    double *values;
    int stencil_indices[nentries];
    values = (double*) calloc(nvalues, sizeof(double));
		
    /* Recall: pi and pj describe position in the processor grid */    
    if (pj == 0) {
      /* Bottom row of grid points */
      double coeff[6];

      coeff_values(coeff, 0., log(rmin), 0., dx0, dx1, dx2, mass, 6);
			
      for (j = 0; j < nvalues; j += nentries) {
	values[j]   = coeff[5] + coeff[2];
	values[j+1] = coeff[0] + coeff[1];
	values[j+2] = coeff[0] - coeff[1];
	values[j+3] = 0.0;
	values[j+4] = 0.0;
	values[j+5] = 0.0;
      }
			
      bc_ilower[0] = pi * ni;
      bc_ilower[1] = pj * nj;
      bc_ilower[2] = pk * nk;
			
      bc_iupper[0] = bc_ilower[0] + ni - 1;
      bc_iupper[1] = bc_ilower[1];
      bc_iupper[2] = bc_ilower[2] + nk - 1;
			
      stencil_indices[0] = 0;
      stencil_indices[1] = 1;
      stencil_indices[2] = 2;
      stencil_indices[3] = 3;
      stencil_indices[4] = 7;
      stencil_indices[5] = 8;
			
      HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
				     stencil_indices, values);
    }

          /*0=a, 1=b, 2=c, 3=d, 4=e, 5=f*/
      /*
	xx 14 xx    10 04 09    xx 18 xx    ^
	11 05 12    01 00 02    15 06 16    |			 
	xx 13 xx    07 03 08    xx 17 xx    j i ->    k - - >			
      */

		
    if (pj == npj - 1) {
      /* upper row of grid points */
      double coeff[6];
			
      bc_ilower[0] = pi * ni;
      bc_ilower[1] = pj * nj + nj - 1;
      bc_ilower[2] = pk * nk;
			
      bc_iupper[0] = bc_ilower[0] + ni - 1;
      bc_iupper[1] = bc_ilower[1];
      bc_iupper[2] = bc_ilower[2] + nk - 1;

      coeff_values(coeff, 0., log(rmax) - dx1, 0., dx0, dx1, dx2, mass, 6);
		  
      for (j = 0; j < nvalues; j += nentries) {
	values[j]   = coeff[5] + coeff[2];
	values[j+1] = coeff[0] - coeff[1];
	values[j+2] = coeff[0] + coeff[1];
	values[j+3] = 0.0;
	values[j+4] = 0.0;
	values[j+5] = 0.0;
      }
			
      stencil_indices[0] = 0;
      stencil_indices[1] = 1;
      stencil_indices[2] = 2;
      stencil_indices[3] = 4;
      stencil_indices[4] = 9;
      stencil_indices[5] = 10;
			
      HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries,
				     stencil_indices, values);
    }
		
    free(values);
  }
  
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
		
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);
		
    HYPRE_StructVectorInitialize(b);
    HYPRE_StructVectorInitialize(x);
		
    /* Set the values */
    for (i = 0; i < nvalues; i++) {
      values[i] = gsl_ran_gaussian(rstate, 1.);
      // TODO possibly give options chosen at compile or run time
    }
    HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);
		
    for (i = 0; i < nvalues; i ++)
      values[i] = 0.0;
    HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);
		
    free(values);
		
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
    /* Get the local raw data */
    int nvalues = ni * nj * nk;
    
    double *raw = (double*)calloc(nvalues, sizeof(double));
    double *env = (double*)calloc(nvalues, sizeof(double));
    
    HYPRE_StructVectorGetBoxValues(x, ilower, iupper, raw);  
  
    /* Find statistics for raw data and envelope */
    double avg_raw = 0.;
    double avg_env = 0.;
    double var_raw = 0.;
    double var_env = 0.;
    double min_raw = raw[0];
    double max_raw = raw[0];
    double min_env = env[0];
    double max_env = env[0];
    
    double *lc_raw = (double*)calloc(npk * nk, sizeof(double));
    double *lc_env = (double*)calloc(npk * nk, sizeof(double));

    /* Calculate mean and variance for raw data */
    {
      for(i = 0; i < nvalues; i++)
	avg_raw += raw[i];
      
      MPI_Allreduce(MPI_IN_PLACE, &avg_raw, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);
      
      avg_raw /= (npi * npj * npk * nvalues);
      
      /* Second pass for variance */
      for (i = 0; i < nvalues; i++)
	var_raw += (raw[i] - avg_raw) * (raw[i] - avg_raw);
      
      MPI_Allreduce(MPI_IN_PLACE, &var_raw, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);
      
      var_raw /= (npi * npj * npk * nvalues - 1);
    }
    
    /* Add envelope and calculate envelope average */
    {
      int l = 0;
      double radius, factor;
      factor = 10.;
      for (k = 0; k < nk; k++) {
	for (j = 0; j < nj; j++) {
	  for (i = 0; i < ni; i++) {
	    radius = rmin * exp( (j + pj * nj) * dx1 )
	      + factor * rmin - rmin;
	    env[l] = 3. * mdot
	      * ( 1. - sqrt( rmin * exp(-1. * dx1) * factor / radius) )
	      / ( 8. * M_PI * pow(radius , 3.) )
	      * exp( 0.5 * (raw[l] - avg_raw) / sqrt(var_raw) );
	      //* (1. + 0.5 * (raw[l] - avg_raw) / sqrt(var_raw) );

	    /* envelope is 3/(8PI)*mdot*GM/r^3*(1-sqrt(r0/r))
	       in units of Msolar*c^2 per r^2 per year, where r = GM/c^2 
	       and c = 1 */
	    //TODO fix units
	    
	    avg_env += env[l];

	    l++;
	  }
	}
      }
      
      MPI_Allreduce(MPI_IN_PLACE, &avg_env, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);

      avg_env /= (npi * npj * npk * nvalues);    
    }
    
    /* Find min, max, lightcurve, var_env */
    {
      int l = 0;
      double area;
      for (k = pk * nk; k < (pk + 1) * nk; k++) {
	for (j = 0; j < nj; j++) {
	  for (i = 0; i < ni; i++) {
	    area = 0.5 * exp( 2. * (j + pj * nj) * dx1 ) *
	      ( exp( 2. * dx1 ) - 1. ) * dx2;
	    lc_raw[k] += raw[l] * area;
	    lc_env[k] += env[l] * area;
	    if (env[l] < min_env)
	      min_env = env[l];
	    if (env[l] > max_env)
	      max_env = env[l];
	    var_env += (env[l] - avg_env) * (env[l] - avg_env);
	    l++;
	  }
	}
      }
      
      MPI_Allreduce(MPI_IN_PLACE, &var_env, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);      

      var_env /= (npi * npj * npk * nvalues - 1);
    }

    // TODO turn allreduce into reduce for min, max, and var
    // (currently required for hdf5_write_single_val)
    MPI_Allreduce(MPI_IN_PLACE, &min_raw, 1, MPI_DOUBLE,
		  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_raw, 1, MPI_DOUBLE,
		  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_env, 1, MPI_DOUBLE,
		  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_env, 1, MPI_DOUBLE,
		  MPI_MAX, MPI_COMM_WORLD);
    if (myid == 0) {
      MPI_Reduce(MPI_IN_PLACE, lc_raw, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, lc_env, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else {
      MPI_Reduce(lc_raw, lc_raw, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(lc_env, lc_env, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
    }

    /* File i/o */
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

      printf("%s\n\n", filename);
    }
    
    MPI_Bcast(&filename, 255, MPI_CHAR, 0, MPI_COMM_WORLD);
    hdf5_create(filename);
    
    /* Save solution to output file*/
    hdf5_set_directory("/");
    hdf5_make_directory("data");
    hdf5_set_directory("/data/");
    // TODO create further heirarchy in file structure

    /* note: HYPRE has k as the slowest varying, opposite of HDF5 */
    {
      hsize_t fdims[3]  = {npk * nk, npj * nj, npi * ni};
      hsize_t fstart[3] = {pk * nk, pj * nj, pi * ni};
      hsize_t fcount[3] = {nk, nj, ni};
      hsize_t mdims[3]  = {nk, nj, ni};
      hsize_t mstart[3] = {0, 0, 0};
      
      hdf5_write_array(raw, "data_raw", 3, fdims, fstart, fcount,
		       mdims, mstart, H5T_NATIVE_DOUBLE);  
      hdf5_write_array(env, "data_env", 3, fdims, fstart, fcount,
		       mdims, mstart, H5T_NATIVE_DOUBLE);
    }
    
    /* Output lightcurve and parameters */
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
	
      hdf5_write_array(lc_raw, "lc_raw", 1, &fdims, &fstart, &fcount,
		       &mdims, &mstart, H5T_NATIVE_DOUBLE);
      hdf5_write_array(lc_env, "lc_env", 1, &fdims, &fstart, &fcount,
		       &mdims, &mstart, H5T_NATIVE_DOUBLE);
    }
    
    hdf5_set_directory("/");
    hdf5_make_directory("params");
    hdf5_set_directory("/params/");

    hdf5_write_single_val(&mass, "mass", H5T_IEEE_F64LE);
    hdf5_write_single_val(&mdot, "mdot", H5T_IEEE_F64LE);
    hdf5_write_single_val(&rmin, "rmin", H5T_IEEE_F64LE);
    hdf5_write_single_val(&rmax, "rmax", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx0, "dt", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx1, "dr", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx2, "dphi", H5T_IEEE_F64LE);

    hdf5_write_single_val(&npi, "npi", H5T_STD_I32LE);
    hdf5_write_single_val(&npj, "npj", H5T_STD_I32LE);
    hdf5_write_single_val(&npk, "npk", H5T_STD_I32LE);
    hdf5_write_single_val(&ni, "ni", H5T_STD_I32LE);
    hdf5_write_single_val(&nj, "nj", H5T_STD_I32LE);
    hdf5_write_single_val(&nk, "nk", H5T_STD_I32LE);
    hdf5_write_single_val(&gsl_rng_default_seed, "seed", H5T_STD_U64LE);
    
    hdf5_set_directory("/");
    hdf5_make_directory("stats");
    hdf5_set_directory("/stats/");
    hdf5_write_single_val(&min_raw, "min_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&max_raw, "max_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&avg_raw, "avg_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&var_raw, "var_raw", H5T_IEEE_F64LE);

    hdf5_write_single_val(&min_env, "min_env", H5T_IEEE_F64LE);
    hdf5_write_single_val(&max_env, "max_env", H5T_IEEE_F64LE);
    hdf5_write_single_val(&avg_env, "avg_env", H5T_IEEE_F64LE);
    hdf5_write_single_val(&var_env, "var_env", H5T_IEEE_F64LE);
    
    hdf5_close();

    check_t = clock();
    if ( (myid == 0) && (timer) )
      printf("Data output: t = %lf\n\n",
	     (double)(check_t - start_t) / CLOCKS_PER_SEC);
    
    free(raw);
    free(env);
    free(lc_raw);
    free(lc_env);
  }
  
  /* Free HYPRE memory */
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
  return 1.;// + log(1. + x1 * log(10.) / (2. * M_PI) );
}

double gam(double x0, double x1, double x2)
{
  return 1.;// * (1. + x1 * log(10.) / (2. * M_PI));
}

double bet1(double x0, double x1, double x2)
{
  return 36.;// * (1. - 0.75 * erf(0.5 * x1 * log(10.) / (2 * M_PI) ));
}

double bet2(double x0, double x1, double x2)
{
  return 16000. * exp(x1);
}

void coeff_values(double* coeff, double x0, double x1, double x2, double dx0,
		  double dx1, double dx2, double mass, int index)
{
  double theta, psi, gamma, beta1, beta2;
  theta = -7. * M_PI / 18.;
  /* phi = arctan(omega) = arctan(sqrt(GM)/r^(3/2))
         = arctan(c^3/(GMe^(3x/2))) since r=GMe^x/c^2)
     c^3/(GM) is canceled out by unit conversion
  */
  psi   = atan( exp( -1.5 * x1 ) );
  gamma = gam(x0, x1, x2);
  beta1 = bet1(x0, x1, x2);
  beta2 = bet2(x0, x1, x2);
  /* dphi^2 */
  coeff[0] = ( gamma + beta1 * sin(theta) * sin(theta)
	       + beta2 * sin(psi) * sin(psi) ) / (dx2 * dx2);
  /* dphidx */
  coeff[1] = 0.5 * beta1 * cos(theta) * sin(theta) / (dx1 * dx2);
  /* dx^2 */
  coeff[2] = ( gamma + beta1 * cos(theta) * cos(theta) ) / (dx1 * dx1);
  /* dphidt */
  coeff[3] = 0.5 * beta2 * cos(psi) * sin(psi) / (dx2 * dx0);
  /* dt^2 */
  coeff[4] = ( gamma + beta2 * cos(psi) * cos(psi) ) / (dx0 * dx0);
  /* const */
  coeff[5] = -2. * ( coeff[0] + coeff[2] + coeff[4] ) - ksq(x0, x1, x2);
}
