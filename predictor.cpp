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

double predictor::PIC_predict(mat &M, vec &b)
{
    vec alpha;
    double rmse = 0.0;
    vm diff(od->nBlock);
    SFOR(i, od->nBlock) diff[i] = vec(this->od->tSize);
    SFOR(i, nz)
    {
        alpha = M * Z.row(i).t() + b;
        mat theta; vec s;
        unvectorise(theta, s, alpha, od->nDim, config->nBasis);

        SFOR(j, od->nBlock)
        {
            mat KBjBj, KTjBj;
            mat XBj = (*od->getxb(j));
            mat XTj = (*od->getxt(j));
            mat YBj = (*od->getyb(j)) - od->y_mean;
            mat YTj = (*od->getyt(j)) - od->y_mean;
            bs->kernel(XBj, theta, od->signal, KBjBj);
            bs->kernel(XTj, XBj, theta, od->signal, KTjBj);

            SFOR(k, od->tSize)
            {
                rowvec xt_jk = od->getxt(j)->row(k);

                mat pred = KTjBj.row(k) * inv(KBjBj + SQR(od->noise) * eye<mat>(od->bSize, od->bSize)) * YBj;

                diff[j][k] = (1.0 / nz) * (YTj(k, 0) - pred(0, 0));
            }
        }
    }

    NFOR(i, j, od->nBlock, od->tSize) rmse += SQR(diff[i](j));
    return sqrt((1.0 / od->nTest) * rmse);
}

double predictor::predict(mat &M, vec &b)
{
    vec alpha;
    double rmse = 0.0;
    vm diff(od->nBlock);
    SFOR(i, od->nBlock) diff[i] = vec(this->od->tSize);
    SFOR(i, nz)
    {
        alpha = M * Z.row(i).t() + b;
        mat theta; vec s;
        unvectorise(theta, s, alpha, od->nDim, config->nBasis);
        NFOR(j, k, od->nBlock, od->tSize)
        {
            rowvec xt_jk = od->getxt(j)->row(k);
            double yt_jk = od->getyt(j)->at(k, 0) - od->y_mean;
            colvec phi;
            bs->Phi(xt_jk, phi, theta);
            diff[j](k) += (1.0 / nz) * (yt_jk - dot(phi, s));
        }
    }
    NFOR(i, j, od->nBlock, od->tSize) rmse += SQR(diff[i](j));

    return sqrt((1.0 / od->nTest) * rmse);
}
