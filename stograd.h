#ifndef STOGRAD_H
#define STOGRAD_H

#include "libhead.h"
#include "basis.h"
#include "organized.h"

class stograd
{
    public:
        stograd(OrganizeData* od);
       ~stograd();

        OrganizedData *od;
        vm dalpha, dvk;
        mat M, dlogqp;
        vec b, vk, rk;
        int nBasis;
        basis* bs;

        void compute_dalpha_dM(vec &z);
        void compute_rk();
        void compute(mat &Z, vec &K, vec &eta, int M_r, int M_c, vec &res);
        void compute_vk(int k, mat &theta, vec &s, vec &vk);
        void compute_dvk(int k, mat &theta, vec &s, vec &dvk);
        void compute_F(mat &theta, vec &s, int k, vec &Fkz);
        void compute_dlogqp(mat &theta, mat &M, vec &s, int k);
};

#endif // STOGRAD_H
