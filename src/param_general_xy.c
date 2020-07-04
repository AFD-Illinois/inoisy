#include "param.h"

#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "hdf5_utils.h"

#define SMALL 1.E-10

const char model_name[] = "general_xy";

/* Grid domain */

const double param_x0start = 0.;         /* t in terms of M */
const double param_x0end   = 100.; 
const double param_x1start = -10.;       /* x in terms of M */
const double param_x1end   = 10.;
const double param_x2start = -10.;       /* y in terms of M */  
const double param_x2end   = 10.;

/* Set default parameters */

/* Mass in solar masses*/
static double param_mass  = 1.E6;
/* Mdot in solar masses per year */
static double param_mdot  = 1.E-7;
/* ratio of correlation length to local radius */
static double param_lam   = 5.;
/* product of correlation time and local Keplerian frequency */
static double param_tau   = 1.;
/* ratio of coefficients of major and minor axes of spatial correlation */
static double param_r12   = 0.1;
/* cutoff radius */
static double param_rct   = 0.5;
/* opening angle, defined as 0 pointing radially outward */
static double param_theta = -M_PI / 2. + M_PI / 9.; 

/* Read in parameters from input file */
void param_read_params(char* filename)
{
  /* hdf5_utils accesses a single global file at a time */
  
  if (filename != NULL) {
    hdf5_open(filename);

    hdf5_set_directory("/params/");
    hdf5_read_single_val(&param_mass, "mass", H5T_IEEE_F64LE);
    hdf5_read_single_val(&param_mdot, "mdot", H5T_IEEE_F64LE);
    hdf5_read_single_val(&param_lam, "lam", H5T_IEEE_F64LE);
    hdf5_read_single_val(&param_tau, "tau", H5T_IEEE_F64LE);
    hdf5_read_single_val(&param_r12, "r12", H5T_IEEE_F64LE);
    hdf5_read_single_val(&param_rct, "rct", H5T_IEEE_F64LE);
    hdf5_read_single_val(&param_theta, "theta", H5T_IEEE_F64LE);
    
    hdf5_close();
  }
}

/* Write out parameters to output file */
void param_write_params(char* filename)
{
  /* hdf5_utils accesses a single global file at a time */
  hdf5_set_directory("/params/");
    
  hdf5_write_single_val(&param_mass, "mass", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_mdot, "mdot", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_lam, "lam", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_tau, "tau", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_r12, "r12", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_rct, "rct", H5T_IEEE_F64LE);
  hdf5_write_single_val(&param_theta, "theta", H5T_IEEE_F64LE);
}

/* Set format for name of output file
   dir is the name of the output directory */
void param_set_output_name(char* filename, int ni, int nj, int nk,
			   int npi, int npj, int npk, char* dir)
{
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[255];

  time(&rawtime);
  timeinfo = localtime(&rawtime);
  
  strftime(buffer, 255, "%Y_%m_%d_%H%M%S", timeinfo);

  sprintf(filename, "%s/%s_%d_%d_%d_%s_%05lu.h5", dir, model_name,
	  npj * nj, npi * ni, npk * nk, buffer, gsl_rng_default_seed);
}

/* smooth cutoff at radius r0, where function has value f(r0) and slope
df(r0). continuous + once differentiable at r0, and has value f(0) and 
slope 0 at r = 0 */
static double cutoff(double r, double r0, double fr0, double dfr0, double f0)
{
  double a, b;
  b = (2. * (fr0 - f0) - r0 * dfr0) / pow( r0, 3. );
  a = (fr0 - f0) / (b * r0 * r0) + r0;
  return b * r * r * (a - r) + f0;
}

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2)
{
  double radius, x, y;

  x = param_x1start + (j + pj * nj) * dx1;
  y = param_x2start + (i + pi * ni) * dx2;
  
  radius = sqrt(x * x + y * y);

  /* double sigma = 5.; */
  /* return exp(-0.5 * radius * radius / (sigma * sigma))  */
  /*   * (raw - avg_raw) / sqrt(var_raw) */
  /*   / ( 2. * M_PI * sigma * sigma); */

  /* if (radius >= param_rct) */
  /*   return 3. * param_mdot * 5.67E46 // solar mass*c^2/year to erg/s */
  /*     * ( 1. - sqrt( param_rct / radius ) ) */
  /*     / (8. * M_PI * pow(radius, 3.) ) */
  /*     * exp( 0.5 * (raw - avg_raw) / sqrt(var_raw) ); */
  /* else */
  /*   return 0.; */
  /* /\* in erg/s per unit area (in M^2) *\/ */

  /* env(r) = (r0 / r)^4 * e^(-r0^2 / r^2)
     env falls off as r^-4 at large r, peak at r = r0 / sqrt(2) with a
     value of 4/e^2 */
  double r0 = 2.;
  /* at r0 / sqrt( log(1/SMALL) )
     env(r) = log(SMALL)^2*SMALL ~ 100 * SMALL */
  if ( radius > r0 / sqrt( log(1. / SMALL) ) ) {
    double ir = r0 / radius;
    return pow(ir, 4.) * exp(-ir * ir)
  	* exp ( 0.5 * (raw - avg_raw) / sqrt(var_raw) );
  }
  else
    return 0.;
}

static double w_keplerian(double x0, double x1, double x2)
{
  double r = sqrt(x1 * x1 + x2 * x2);
  
  if (r >= param_rct)
    return pow(r, -1.5);
  else
    return cutoff(r, param_rct, pow(param_rct, -1.5),
		  -1.5 * pow(param_rct, -2.5), 0.9 * pow(param_rct, -1.5));
}

static double corr_length(double x0, double x1, double x2)
{
  /* return param_lam; */
  
  double r = sqrt(x1 * x1 + x2 * x2);
  
  if (r >= param_rct)
    return param_lam * r;
  else
    return cutoff(r, param_rct, param_lam * param_rct,
		  param_lam, 0.9 * param_lam * param_rct);
}

static double corr_time(double x0, double x1, double x2)
{
  /* return param_tau; */
  
  double r = sqrt(x1 * x1 + x2 * x2);

  if (r >= param_rct)
    return 2. * M_PI * param_tau / fabs( w_keplerian(x0, x1, x2) );
  else
    return cutoff(r, param_rct,
		  2. * M_PI * param_tau * pow(param_rct, 1.5),
		  2. * M_PI * param_tau * 1.5 * sqrt(param_rct),
		  0.9 * 2. * M_PI * param_tau * pow(param_rct, 1.5) );
}

/* time correlation vector (1, v1, v2) */
static void set_u0(double* u0, double x0, double x1, double x2)
{
  double omega = w_keplerian(x0, x1, x2);
  u0[0] = 1.;
  u0[1] = -x2 * omega;
  u0[2] = x1 * omega;

  //  u0[1] = sin( x2 * 2. * M_PI / (param_x2end - param_x2start) );
  /* u0[1] = 0.; */
  /* u0[2] = 0.; */
}

/* unit vectors in direction of major and minor axes */
static void set_u1_u2(double* u1, double* u2, double x0, double x1, double x2)
{
  double theta;
  
  u1[0] = 0.;
  u2[0] = 0.;
  
  if (x1 == 0 && x2 == 0) {
    u1[1] = 0.;
    u1[2] = 0.;
    
    u2[1] = 0.;
    u2[2] = 0.;
  }
  else {
    theta = atan2(x2, x1) +
      copysign( param_theta, -1. * w_keplerian(x0, x1, x2) );
    
    /* double dx0 = (param_x0end - param_x0start) / 512.; */
    /* theta = atan2(1, dx0 * 2. * M_PI * */
    /* 		  cos(x2 * 2. * M_PI / (param_x2end - param_x2start) ) */
    /* 		  / (param_x2end - param_x2start) ); */
    
    u1[1] = cos(theta);
    u1[2] = sin(theta);
    
    u2[1] = -sin(theta);
    u2[2] = cos(theta);
  }

  /* if (x1 > 2. * x2 + 10) */
  /*   theta = atan(0.5); */
  /* else if (x1 > -2. * x2 + 10) */
  /*   theta = atan(-0.5); */
  /* else */
  /*   theta = M_PI / 2.; */
  
  /* u1[1] = cos(theta); */
  /* u1[2] = sin(theta); */

  /* u2[1] = -sin(theta); */
  /* u2[2] = cos(theta); */
}

static void set_h(double h[3][3], double x0, double x1, double x2)
{
  int i, j;
  double u0[3], u1[3], u2[3];

  set_u0(u0, x0, x1, x2);
  set_u1_u2(u1, u2, x0, x1, x2);
  
  double lam0, lam1, lam2; /* temporal, major, minor correlation lengths */

  lam0 = corr_time(x0, x1, x2);
  lam1 = corr_length(x0, x1, x2);
  lam2 = param_r12 * lam1;
  
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      h[i][j] = lam0 * lam0 * u0[i] * u0[j]
	+ lam1 * lam1 * u1[i] * u1[j]
	+ lam2 * lam2 * u2[i] * u2[j];
    }
  }
}

/* dh[0][2][1] = dh[0][2]/dx[1] */
static void set_dh(double dh[][3][3], double x0, double x1, double x2,
		   double dx0, double dx1, double dx2)
{
  int i, j, k;

  double dx[3] = {dx0, dx1, dx2};
  
  /* hm[0][2][1] = h(x0 - dx0, x1, x2)[2][1] */
  double hm[3][3][3], hp[3][3][3]; 
  set_h(hm[0], x0 - dx0, x1, x2);
  set_h(hp[0], x0 + dx0, x1, x2);
  set_h(hm[1], x0, x1 - dx1, x2);
  set_h(hp[1], x0, x1 + dx1, x2);
  set_h(hm[2], x0, x1, x2 - dx2);
  set_h(hp[2], x0, x1, x2 + dx2);

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++)
	dh[i][j][k] = 0.5 * ( hp[k][i][j] - hm[k][i][j] ) / dx[k];
}

static double ksq(double x0, double x1, double x2)
{
  return 1.;
  //  return param_tau * param_r02 + param_r12 * corr_length(x0, x1, x2);
  // + log(1. + x1 * log(10.) / (2. * M_PI) );
}

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0,
		  double dx1, double dx2, int index)
{
  double h[3][3], dh[3][3][3];
  set_h(h, x0, x1, x2);
  set_dh(dh, x0, x1, x2, dx0, dx1, dx2);
  
  /* dy^2 */
  coeff[0] = -h[2][2] / (dx2 * dx2);
  /* dydx */
  coeff[1] = -0.5 * h[1][2] / (dx1 * dx2);
  /* dx^2 */
  coeff[2] = -h[1][1] / (dx1 * dx1);
  /* dydt */
  coeff[3] = -0.5 * h[0][2] / (dx0 * dx2);
  /* dxdt */
  coeff[4] = -0.5 * h[0][1] / (dx0 * dx1);
  /* dt^2 */
  coeff[5] = -h[0][0] / (dx0 * dx0);
  /* dy */
  coeff[6] = -0.5 * ( dh[0][2][0] + dh[1][2][1] + dh[2][2][2] ) / dx2;
  /* dx */
  coeff[7] = -0.5 * ( dh[0][1][0] + dh[1][1][1] + dh[2][1][2] ) / dx1;
  /* dt */
  coeff[8] = -0.5 * ( dh[0][0][0] + dh[1][0][1] + dh[2][0][2] ) / dx0;
  /* const */
  coeff[9] = -2. * ( coeff[0] + coeff[2] + coeff[5] ) + ksq(x0, x1, x2);
}

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk,
		      int pi, int pj, int pk, int npi, int npj, int npk,
		      double dx0, double dx1, double dx2, int nrecur)
{
  int i;
  int nvalues = ni * nj * nk;

  double x0, x1, x2;

  int gridi, gridj, gridk;

  /* White noise scaled by N
     N = sqrt( sqrt(dL) * (4 Pi)^(D/2) * Gamma(alpha) / Gamma(nu)
     dL = det Lambda = l0^2*l1^2*l2^2 */
  
  double scaling = pow(4. * M_PI, 3. / 2.) * tgamma(2. * nrecur)
    / tgamma(2. * nrecur - 3. / 2. ); 
  
  for (i = 0; i < nvalues; i++) {
    gridk = i / (ni * nj);
    gridj = (i - ni * nj * gridk) / ni;
    gridi = i - ni * nj * gridk + (pi - gridj) * ni;
    gridj += pj * nj;
    gridk += pk * nk;

    x0 = param_x0start + dx0 + gridk;
    x1 = param_x1start + dx1 * gridj;
    x2 = param_x2start + dx2 * gridi;

    //    double r = sqrt(x1 * x1 + x2 * x2);

    scaling *= corr_time(x0, x1, x2) * corr_length(x0, x1, x2)
      * param_r12 * corr_length(x0, x1, x2);
    scaling = fmax( sqrt(scaling), SMALL );
    
    values[i] = gsl_ran_gaussian_ziggurat(rstate, 1.) * scaling;
  }
}
