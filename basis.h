#ifndef BASIS_H
#define BASIS_H

#include "libhead.h"

class basis
{
    public:
        basis();
       ~basis();

        void Phi(rowvec &x, colvec &phi, mat &theta);
        // compute the feature vector phi = Phi(x) given the parameter theta = [theta_1 theta_2 ... theta_m]
        // where Phi(x) = [cos(2 * pi * x * theta_1) sin(2 * pi * x * theta_1) ... cos(2 * pi * x * theta_m) sin(2 * pi * x * theta_m)]'
        void dPhi_dtheta(rowvec &x, mat &theta, colvec &phi, mat &dphi);
        // compute dphi(x) = [dcos(2 * pi * x * theta_1)/dtheta_1 dsin(2 * pi * x * theta_1)/dtheta_1 ... dcos(2 * pi * x * theta_m)/dtheta_m dsin(2 * pi * x * theta_m)/dtheta_m]
        // given x, theta = [theta_1 theta_2 ... theta_m] and a precomputed phi = Phi(x)
        void dPhi_dtheta(rowvec &x, mat &theta, mat &dphi);
        // compute dphi(x) = [dcos(2 * pi * x * theta_1)/dtheta_1 dsin(2 * pi * x * theta_1)/dtheta_1 ... dcos(2 * pi * x * theta_m)/dtheta_m dsin(2 * pi * x * theta_m)/dtheta_m]
        // given x, theta = [theta_1 theta_2 ... theta_m]
        double kernel(rowvec &a, rowvec &b, mat &theta, double signal);
        double kernel(rowvec &a, mat &theta, double signal);
        void   kernel(mat &A, mat &B, mat &theta, double signal, mat &KAB);
        void   kernel(mat &A, mat &theta, double signal, mat &KAA);
};

#endif // BASIS_H
