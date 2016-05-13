#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "libhead.h"
#include "organized.h"
#include "basis.h"
#include "configuration.h"

class predictor
{
    public:

        predictor(int nz, vec &zmean, mat &zcov, OrganizedData* od, configuration* config);
       ~predictor();

        int nz;
        vec zmean;
        mat zcov, Z;
        basis *bs;

        OrganizedData* od;
        configuration* config;

        pdd combined_predict(mat &M, vec &b);
        double fast_predict(mat &M, vec &b);
        double fast_predict(mat &M, vec &b, rowvec &xt, double yt, bmat &cov_matrices);
        double pro_predict(mat &M, vec &b, rowvec &xt, double yt);
        double pro_predict(mat &M, vec &b);
        double predict(mat &M, vec &b);
        double PIC_predict(mat &M, vec &b);
};

#endif // PREDICTOR_H
