#include "predictor.h"

predictor::predictor(int nz, vec &zmean, mat &zcov, OrganizedData* od, configuration* config)
{
    this->nz     = nz;
    this->zmean  = zmean;
    this->zcov   = zcov;
    this->od     = od;
    this->config = config;

    Z = mat(nz, zmean.n_rows);
    multi_gaussian_random(zmean, zcov, nz, Z);
    bs = new basis();
}

predictor::~predictor()
{
    //dtor
}

double predictor::predict(mat &M, vec &b)
{
    vec alpha;
    double rmse = 0.0;
    vm diff(od->nBlock);
    SFOR(i, od->nBlock) diff[i] = vec(tSize);
    SFOR(i, nz)
    {
        alpha = M * Z.row(i).t() + b;
        mat theta;
        vec s;
        unvectorise(theta, s, alpha, od->nDim, config->nBasis);
        NFOR(j, k, od->nBlock, od->tSize)
        {
            vec xt_jk = od->getxt(j)->row(k);
            double yt_jk = od->getyt(j)->at(k, 0);
            vec phi;
            bs->Phi(xt_jk, phi, theta);
            diff[j](k) += (1.0 / nz) * (yt_jk - dot(phi, s));
        }
    }
    NFOR(i, j, od->nBlock, od->tSize) rmse += SQR(diff[i](j));

    return sqrt((1.0 / od->nTest) * rmse);
}
