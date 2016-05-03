#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "libhead.h"
#include "organized.h"
#include "stograd.h"
#include "basis.h"
#include "configuration.h"
#include "predictor.h"

#define DEFAULT_NO_SAMPLE 5

class optimizer
{
    public:

        optimizer(OrganizedData* od, configuration* config, int nBasis);
       ~optimizer();

        int nBasis;
        OrganizedData* od;
        configuration* config;
        stograd* gradient;
        predictor* model;

        vec eta, b, z_mean, db;
        mat M, z_cov, dM;
        field <mat> savedItems;

        time_t t1, t2;

        void initialize();
        pair <double, double> learning_rate(int nIter);
        void optimize();
};

#endif // OPTIMIZER_H
