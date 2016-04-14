#include "basis.h"


basis::basis()
{
    //ctor
}

basis::~basis()
{
    //dtor
}

void basis::Phi(rowvec &x, colvec &phi, mat &theta)
{
    phi = colvec(2 * theta.n_cols);

    SFOR(i, theta.n_cols)
    {
        double dp       = dot(theta.col(i), x);
        phi(2 * i)      = cos(2 * PI * dp);
        phi(2 * i + 1)  = sin(2 * PI * dp);
    }
}

void basis::dPhi_dtheta(rowvec &x, mat &theta, colvec &phi, mat &dphi)
{
    dphi = zeros <mat> (x.n_elem, 2 * theta.n_cols);

    SFOR(i, theta.n_cols)
    {
        dphi.col(2 * i)     = - 2 * PI * phi(2 * i + 1) * x.t();
        dphi.col(2 * i + 1) = 2   * PI * phi(2 * i)     * x.t();
    }
}

void basis::dPhi_dtheta(rowvec &x, mat &theta, mat &dphi)
{
    vec phi; Phi(x, phi, theta);
    dPhi_dtheta(x, theta, phi, dphi);
}
