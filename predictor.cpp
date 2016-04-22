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
    double rmse = 0.0;

    int nThread = omp_get_num_procs();

    vector <vm> diff(nThread);
    SFOR(i, nThread)
    {
        diff[i] = vm(od->nBlock);
        SFOR(j, od->nBlock) diff[i][j] = zeros <mat> (od->tSize, 1);
    }

    int chunk = nz / nThread;
	if (chunk == 0) chunk++;

	#pragma omp parallel for schedule(dynamic, chunk)
    SFOR(i, nz)
    {
        int t = omp_get_thread_num();
        vec alpha = M * Z.row(i).t() + b;
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

            KBjBj.diag() += SQR(od->noise);
            mat pred = KTjBj * inv_sympd(KBjBj) * YBj;
            diff[t][j] += (1.0 / nz) * (YTj - pred);
            /*SFOR(k, od->tSize)
                diff[t][j][k] += (1.0 / nz) * (YTj(k, 0) - pred(k, 0));*/

            KBjBj.clear(); KTjBj.clear(); pred.clear();
            XBj.clear(); YBj.clear();
            XTj.clear(); YTj.clear();
        }
        theta.clear(); alpha.clear(); s.clear();
    }

    NFOR(i, j, od->nBlock, od->tSize)
    {
        double sum = 0.0;
        SFOR(t, nThread) sum += diff[t][i][j];
        rmse += SQR(sum);
    }

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
