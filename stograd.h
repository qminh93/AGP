#ifndef STOGRAD_H
#define STOGRAD_H

#include "libhead.h"
#include "basis.h"
#include "organized.h"

class stograd
{
    public:
        stograd(OrganizedData* od, int nBasis);
       ~stograd();

        OrganizedData *od;
        mat M, M_inv;
        vec b;
        int nBasis;
        basis* bs;

        void compute_dalpha_dM(vec &z, vm &dalpha);
        void compute_rk(vec &vk, vm &dvk, vec &rk);
        void compute(mat &Z, vec &K, vec &eta, int M_r, int M_c, vec &res);
        void compute_vk(int k, mat &theta, vec &s, vec &vk);
        void compute_dvk(int k, mat &theta, vec &s, vm &dalpha, vm &dvk);
        void compute_F(mat &theta, vec &s, int k, vm &dalpha, vec &Fkz);
        void compute_dlogqp(mat &theta, vec &s, vm &dalpha, vec &dlogqp);
};

#endif // STOGRAD_H
