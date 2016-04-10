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
        mat M;
        vec b;
        int nBasis;
        basis* bs;

        void compute(mat &Z, vec &K, vec &eta, int M_r, int M_c, vec &res);
        void compute_vk(int k, mat &theta, vec &s, vec &vk);
        void compute_dvk_dalpha(int k, mat &theta, vec &s, vec &dvk);
        void compute_F(mat &theta, vec &s, int k, vec &Fkz);
        void compute_dlogqp_dalpha(mat &theta, vec &s, int k, vec &logqp);
};

#endif // STOGRAD_H
