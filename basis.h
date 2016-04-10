#ifndef BASIS_H
#define BASIS_H

#include "libhead.h"

class basis
{
    public:
        basis();
       ~basis();

        void Phi(rowvec &x, colvec &phi, mat &theta);
        void dPhi_dtheta(rowvec &x, mat &theta, colvec &phi, mat &dphi);
        void dPhi_dtheta(rowvec &x, mat &theta, mat &dphi);
};

#endif // BASIS_H
