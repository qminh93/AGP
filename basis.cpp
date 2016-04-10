#include "basis.h"

basis::basis(mat &Theta)
{
    this.Theta = Theta;
}

basis::~basis()
{
    //dtor
}

void basis::Phi(rowvec &x, colvec &phi)
{
    phi = colvec(2 * Theta.n_rows, 0);

    SFOR(i, Theta.n_rows)
    {
        double dp       = dot(Theta.row(i), x);
        phi(2 * i)      = cos(2 * PI * dp);
        phi(2 * i + 1)  = sin(2 * PI * dp);
    }
}
