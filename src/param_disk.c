#include "param.h"

#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

const double param_mass    = 1.E6; /* in solar masses*/
const double param_mdot    = 1.E-7; /* in solar masses per year */
const double param_x0start = 0.; /* t in terms of M */
const double param_x0end   = 5000.; 
const double param_x1start = log(2.); /* x1 = log(r), r in terms of M */
const double param_x1end   = log(100.);
const double param_x2start = 0.;
const double param_x2end   = 2. * M_PI;

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2)
{
  double radius, factor;
  factor = 10.;
  radius = param_x1start * exp( (j + pj * nj) * dx1 )
    + factor * param_x1start - param_x1start;
  
  return 3. * param_mdot
    * ( 1. - sqrt( param_x1start * exp(-1. * dx1) * factor / radius) )
    / ( 8. * M_PI * pow(radius , 3.) )
    * exp( 0.5 * (raw - avg_raw) / sqrt(var_raw) );
  // * (1. + 0.5 * (raw[l] - avg_raw) / sqrt(var_raw) ); 
  
  /* envelope is 3/(8PI)*mdot*GM/r^3*(1-sqrt(r0/r))
     in units of Msolar*c^2 per r^2 per year, where r = GM/c^2 
     and c = 1 */
  //TODO fix units
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

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0,
		  double dx1, double dx2, int index)
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

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk,
		      int pi, int pj, int pk, int npi, int npj, int npk,
		      double dx0, double dx1, double dx2)
{
  int i;
  int nvalues = ni * nj * nk;

  for (i = 0; i < nvalues; i++) {
    values[i] = gsl_ran_gaussian_ziggurat(rstate, 1.);
  }
}
