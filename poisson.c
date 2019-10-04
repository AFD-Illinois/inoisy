/*
   Compile with:   make poisson

   Sample run:     mpirun -np 4 poisson -n 33 -solver 0
	                 mpiexec -n 4 ./poisson -n 33 -solver 0

   To see options: poisson -help
*/

#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"

double ksq(double x0, double x1, double x2);

double gam(double x0, double x1, double x2);

double bet1(double x0, double x1, double x2);

double bet2(double x0, double x1, double x2);

void coeff_values(double* coeff, double x0, double x1, double x2, double dx, double dy, double dz, int index);

int main (int argc, char *argv[])
{
	int i, j, k;
	
	int myid, num_procs;
	
	int n, N, pi, pj, pk;
	double dx, dy, dz;
	int ilower[3], iupper[3];
	
	int solver_id;
	int n_pre, n_post;
	
	HYPRE_StructGrid     grid;
	HYPRE_StructStencil  stencil;
	HYPRE_StructMatrix   A;
	HYPRE_StructVector   b;
	HYPRE_StructVector   x;
	HYPRE_StructSolver   solver;
	HYPRE_StructSolver   precond;
	
	int num_iterations;
	double final_res_norm;
	
	int vis;
	
	/* Initialize MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	
	/* Set defaults */
	n = 32;
	solver_id = 0;
	n_pre  = 1;
	n_post = 1;
	vis = 0;
	
	const gsl_rng_type *T;
	gsl_rng *rstate;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	rstate = gsl_rng_alloc(T);
	gsl_rng_set(rstate, gsl_rng_default_seed + myid);	 
	
	/* Parse command line */
	{
		int arg_index = 0;
		int print_usage = 0;
		
		while (arg_index < argc) {
			if ( strcmp(argv[arg_index], "-n") == 0 ) {
				arg_index++;
				n = atoi(argv[arg_index++]);
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
			else if ( strcmp(argv[arg_index], "-vis") == 0 ) {
				arg_index++;
				vis = 1;
			}
			else if ( strcmp(argv[arg_index], "-help") == 0 ) {
				print_usage = 1;
				break;
			}
			else {
				arg_index++;
			}
		}
		
		if ((print_usage) && (myid == 0)) {
			printf("\n");
			printf("Usage: %s [<options>]\n", argv[0]);
			printf("\n");
			printf("  -n <n>              : problem size per processor (default: 33)\n");
			printf("  -solver <ID>        : solver ID\n");
			printf("                        0  - PCG with SMG precond (default)\n");
			printf("                        1  - SMG\n");
			printf("  -v <n_pre> <n_post> : number of pre and post relaxations (default: 1 1)\n");
			printf("  -vis                : save the solution for GLVis visualization\n");
			printf("\n");
		}
		
		if (print_usage) {
			MPI_Finalize();
			return (0);
		}
	}
	
	/* Figure out the processor grid (N x N x 1).  The local problem
		 size for the interior nodes is indicated by n (n x n x n).
		 pi and pj and pk indicate position in the processor grid. */
	N  = sqrt(num_procs);
	dx = 2. * M_PI / (N*n); 
	dy = 2. * M_PI / (N*n);
	dz = 2. * M_PI / (n);
	pj = myid / N;
	pi = myid - pj*N;
	pk = 0;
	
	/* Figure out the extents of each processor's piece of the grid. */
	ilower[0] = pi*n;
	ilower[1] = pj*n;
	ilower[2] = pk*n;
	
	iupper[0] = ilower[0] + n-1;
	iupper[1] = ilower[1] + n-1;
	iupper[2] = ilower[2] + n-1;
	
	/* 1. Set up a grid */
	{
		/* Create an empty 3D grid object */
		HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid);
		
		/* Add a new box to the grid */
		HYPRE_StructGridSetExtents(grid, ilower, iupper);

		/* Set periodic boundary conditions on t and phi*/
		int boundcon[3] = {N*n, N*n, n};
		HYPRE_StructGridSetPeriodic(grid, boundcon);
		
		/* This is a collective call finalizing the grid assembly.
			 The grid is now ``ready to be used'' */
		HYPRE_StructGridAssemble(grid);
	}
	
	/* 2. Define the discretization stencil */
	{
		
#define NSTENCIL 27
		
		/* Create an empty 3D, 15-pt stencil object */
		HYPRE_StructStencilCreate(3, NSTENCIL, &stencil);
		
		/* Define the geometry of the stencil */
		/*
			22 14 21    10 04 09    26 18 25    ^
			11 05 12    01 00 02    15 06 16    |
			19 13 20    07 03 08    23 17 24    j i ->    k - - >
			
		  Delete zero entries:
	  	xx 12 xx    10 04 09    xx 14 xx    ^
 	 	  xx 05 xx    01 00 02    xx 06 xx    |			 
	 	  xx 11 xx    07 03 08    xx 13 xx    j i ->    k - - >
			
			13,14 -> 11,12; 17,18 -> 13,14
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
																	{0,-1,1}, {0,1,1},
																	{-1,-1,-1}, {1,-1,-1},
																	{1,1,-1}, {-1,1,-1},
																	{-1,-1,1}, {1,-1,1},
																	{1,1,1}, {-1,1,1}
			}; 
			
			for (entry = 0; entry < NSTENCIL; entry++)
				HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
		}
	}
	
	/* 3. Set up a Struct Matrix */
	{
		int nentries = NSTENCIL;
		int nvalues = nentries*n*n*n;
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
			gridk = temp / (n * n);
			gridj = (temp - n * n * gridk) / n;
			gridi = temp - n * n * gridk + (pi - gridj) * n;
			gridj += pj * n;
			gridk += pk * n;
			
			x0 = dx * gridi;
			x1 = dy * gridj;
			x2 = dz * gridk;
			
			coeff_values(coeff, x0, x1, x2, dx, dy, dz, 6);
			
			/*
				theta = 0.;
				gamma = -1.;
				beta = 0.;
				ast = gamma + beta * cos(theta) * cos(theta);
				bst = 0.5 * beta * cos(theta) * sin(theta);
				cst = gamma + beta * sin(theta) * sin(theta);
				dst = -2. * ( ast + cst + 0. * h2 );
			*/
			
			/*0=a, 1=b, 2=c, 3=d, 4=e, 5=f*/
			/*
				xx 12 xx    10 04 09    xx 14 xx    ^
				xx 05 xx    01 00 02    xx 06 xx    |			 
				xx 11 xx    07 03 08    xx 13 xx    j i ->    k - - >
			*/
			values[i]    = -6.;
			values[i+1]  = 1.;
			values[i+2]  = 1.;
			values[i+3]  = 1.;
			values[i+4]  = 1.;
			values[i+5]  = 1.;
			values[i+6]  = 1.;
			values[i+7]  = 0.;
			values[i+8]  = 0.;
			values[i+9]  = 0.;
			values[i+10] = 0.;
			values[i+13] = 0.;
			values[i+14] = 0.;
			values[i+17] = 0.;
			values[i+18] = 0.;
			
			values[i+11] = 0.;
			values[i+12] = 0.;
			values[i+15] = 0.;
			values[i+16] = 0.;
			values[i+19] = 0.;
			values[i+20] = 0.;
			values[i+21] = 0.;
			values[i+22] = 0.;
			values[i+23] = 0.;
			values[i+24] = 0.;
			values[i+25] = 0.;
			values[i+26] = 0.;
		}

		HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries,
																	 stencil_indices, values);
		
		free(values);
	}
 	
	/* /\* 4. Incorporate the boundary conditions: go along each edge of */
	/* 	 the domain and set the stencil entry that reaches to the boundary.*\/ */
	/* { */
	/* 	int bc_ilower[3]; */
	/* 	int bc_iupper[3]; */
	/* 	int nentries = 3; */
	/* 	int nvalues  = nentries*n*n; /\*  number of stencil entries times the length */
	/* 																 of one side of my grid box *\/ */
	/* 	double *values; */
	/* 	int stencil_indices[3]; */
	/* 	values = (double*) calloc(nvalues, sizeof(double)); */
		
	/* 	/\* Recall: pi and pj describe position in the processor grid *\/ */
	/* 	if (pi == 0) { */
	/* 		/\* Bottom row of grid points *\/ */
	/* 		for (j = 0; j < nvalues; j+=nentries) { */
	/* 			values[j]   = 0.0; */
	/* 			values[j+1] = 0.0; */
	/* 			values[j+2] = 0.0; */
	/* 		} */
			
	/* 		bc_ilower[0] = pi*n; */
	/* 		bc_ilower[1] = pj*n; */
	/* 		bc_ilower[2] = pk*n; */
			
	/* 		bc_iupper[0] = bc_ilower[0]; */
	/* 		bc_iupper[1] = bc_ilower[1] + n-1; */
	/* 		bc_iupper[2] = bc_ilower[2] + n-1; */
			
	/* 		stencil_indices[0] = 1; */
	/* 		stencil_indices[1] = 7; */
	/* 		stencil_indices[2] = 10; */
			
	/* 		HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries, */
	/* 																	 stencil_indices, values); */
	/* 	} */
		
	/* 	if (pi == N-1) { */
	/* 		/\* upper row of grid points *\/ */
	/* 		bc_ilower[0] = pi*n + n-1; */
	/* 		bc_ilower[1] = pj*n; */
	/* 		bc_ilower[2] = pk*n; */
			
	/* 		bc_iupper[0] = bc_ilower[0]; */
	/* 		bc_iupper[1] = bc_ilower[1] + n-1; */
	/* 		bc_iupper[2] = bc_ilower[2] + n-1; */

	/* 		for (j = 0; j < nvalues; j+=nentries) { */
	/* 			values[j]   = 0.0; */
	/* 			values[j+1] = 0.0; */
	/* 			values[j+2] = 0.0; */
	/* 		} */
			
	/* 		stencil_indices[0] = 2; */
	/* 		stencil_indices[1] = 8; */
	/* 		stencil_indices[2] = 9; */
			
	/* 		HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries, */
	/* 																	 stencil_indices, values); */
	/* 	} */
		
	/* 	free(values); */
	/* } */

	/* 4. Incorporate/*  the boundary conditions: go along each edge of */
	/* 	 the domain and set the stencil entry that reaches to the boundary.*\/ */
	/* { */
	/* 	int bc_ilower[3]; */
	/* 	int bc_iupper[3]; */
	/* 	int	nentries = 6; */
	/* 	int nvalues  = nentries*n*n; /\*  number of stencil entries times the length */
	/* 																 of one side of my grid box *\/ */
	/* 	double *values; */
	/* 	int stencil_indices[nentries]; */
	/* 	values = (double*) calloc(nvalues, sizeof(double)); */
		
	/* 	/\* Recall: pi and pj describe position in the processor grid *\/ */
	/* 	if (pi == 0) { */
	/* 		/\* Bottom row of grid points *\/ */
	/* 		double coeff[6]; */

	/* 		coeff_values(coeff, 0., 0., 0., dx, dy, dz, 6); */
			
	/* 		for (j = 0; j < nvalues; j+=nentries) { */
	/* 			values[j]   = coeff[5]+coeff[0]; */
	/* 			values[j+1] = 0.0; */
	/* 			values[j+2] = coeff[2]+coeff[1]; */
	/* 			values[j+3] = coeff[2]-coeff[1]; */
	/* 			values[j+4] = 0.0; */
	/* 			values[j+5] = 0.0; */
	/* 		} */
			
	/* 		bc_ilower[0] = pi*n; */
	/* 		bc_ilower[1] = pj*n; */
	/* 		bc_ilower[2] = pk*n; */
			
	/* 		bc_iupper[0] = bc_ilower[0]; */
	/* 		bc_iupper[1] = bc_ilower[1] + n-1; */
	/* 		bc_iupper[2] = bc_ilower[2] + n-1; */
			
	/* 		stencil_indices[0] = 0; */
	/* 		stencil_indices[1] = 1; */
	/* 		stencil_indices[2] = 3; */
	/* 		stencil_indices[3] = 4; */
	/* 		stencil_indices[4] = 7; */
	/* 		stencil_indices[5] = 10; */
			
	/* 		HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries, */
	/* 																	 stencil_indices, values); */
	/* 	} */
		
	/* 	if (pi == N-1) { */
	/* 		/\* upper row of grid points *\/ */
	/* 		double coeff[6]; */
			
	/* 		bc_ilower[0] = pi*n + n-1; */
	/* 		bc_ilower[1] = pj*n; */
	/* 		bc_ilower[2] = pk*n; */
			
	/* 		bc_iupper[0] = bc_ilower[0]; */
	/* 		bc_iupper[1] = bc_ilower[1] + n-1; */
	/* 		bc_iupper[2] = bc_ilower[2] + n-1; */

	/* 		coeff_values(coeff, bc_ilower[0] * dx, 0., 0., dx, dy, dz, 6); */
		  
	/* 		for (j = 0; j < nvalues; j+=nentries) { */
	/* 			values[j]   = coeff[5]+coeff[0]; */
	/* 			values[j+1] = 0.0; */
	/* 			values[j+2] = coeff[2]-coeff[1]; */
	/* 			values[j+3] = coeff[2]+coeff[1]; */
	/* 			values[j+4] = 0.0; */
	/* 			values[j+5] = 0.0; */
	/* 		} */
			
	/* 		stencil_indices[0] = 0; */
	/* 		stencil_indices[1] = 2; */
	/* 		stencil_indices[2] = 3; */
	/* 		stencil_indices[3] = 4; */
	/* 		stencil_indices[4] = 8; */
	/* 		stencil_indices[5] = 9; */
			
	/* 		HYPRE_StructMatrixSetBoxValues(A, bc_ilower, bc_iupper, nentries, */
	/* 																	 stencil_indices, values); */
	/* 	} */
		
	/* 	free(values); */
	/* } */
	
	/* This is a collective call finalizing the matrix assembly.
		 The matrix is now ``ready to be used'' */
	HYPRE_StructMatrixAssemble(A);
	
	/* 5. Set up Struct Vectors for b and x */
	{
		int    nvalues = n*n*n;
		double *values;
		
		values = (double*) calloc(nvalues, sizeof(double));
		
		/* Create an empty vector object */
		HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
		HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);
		
		/* Indicate that the vector coefficients are ready to be set */
		HYPRE_StructVectorInitialize(b);
		HYPRE_StructVectorInitialize(x);
		
		/* Set the values */
		for (i = 0; i < nvalues; i ++) {
			values[i] = -8. * M_PI * M_PI * sin(4. * M_PI * (double)(i/(n*n)) / n) * sin(4. * M_PI * (double)(i % n) / n) * pow(2./n, 2.);
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

	/* Output data */
	if (vis) {
		FILE *file;
		char filename[255];
		
		int nvalues = n*n*n;
		double *values = (double*) calloc(nvalues, sizeof(double));
		
		/* get the local solution */
		HYPRE_StructVectorGetBoxValues(x, ilower, iupper, values);
		
		sprintf(filename, "%s.%06d", "vis/ex3.sol", myid);
		if ((file = fopen(filename, "w")) == NULL) {
			printf("Error: can't open output file %s\n", filename);
			MPI_Finalize();
			exit(1);
		}
		
		/* save solution with global unknown numbers */
		int l = 0;
		for (k = 0; k < n; k++)
			for (j = 0; j < n; j++)
				for (i = 0; i < n; i++)
					fprintf(file, "%06d %.14e\n", pk*N*n*n*n+pj*N*n*n+pi*n+k*N*n*n+j*N*n+i, values[l++]);
		
		fflush(file);
		fclose(file);
		free(values);		
	}
	
	if (myid == 0) {
		printf("\n");
		printf("Iterations = %d\n", num_iterations);
		printf("Final Relative Residual Norm = %g\n", final_res_norm);
		printf("\n");
	}
	
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
	return 1.;// + log(1. + x0);
}

double gam(double x0, double x1, double x2)
{
	return 1.;// * (1. + x0 / (2. * M_PI));
}

double bet1(double x0, double x1, double x2)
{
	return 36.;// * (1. - 0.75 * erf(0.5*x0));
}

double bet2(double x0, double x1, double x2)
{
	return 100.;
}

void coeff_values(double* coeff, double x0, double x1, double x2, double dx, double dy, double dz, int index)
{
	double theta, psi, gamma, beta1, beta2;
	theta = -7. * M_PI / 18.;
	psi = atan( exp( 4. - 1.5 * ( 2. * dx + x0 / 2. ) ) );
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
