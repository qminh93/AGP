#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "libhead.h"
#include "organized.h"
#include "basis.h"

class predictor
{
    public:

        predictor(int nz, vec &zmean, mat &zcov, OrganizedData* od, configuration* config);
       ~predictor();

        int nz;
        vec zmean;
        mat zcov, Z;
        basis bs;

        double predict(mat &M, vec &b);
};

#endif // PREDICTOR_H
