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

double predictor::fast_predict(mat &M, vec &b)
{
    bmat cov_matrices(nz, od->nBlock);

    int nThread = omp_get_num_procs(), chunk = nz / nThread;
	if (chunk == 0) chunk++;

	#pragma omp parallel for schedule(dynamic, chunk)
    SFOR(i, nz)
    {
        vec alpha = M * Z.row(i).t() + b;
        mat theta; vec s;
        unvectorise(theta, s, alpha, od->nDim, config->nBasis);

        SFOR(j, od->nBlock)
        {
            //cout << i << " " << j << endl;
            mat XBj = (*od->getxb(j)), KBjBj;
            bs->kernel(XBj, theta, od->signal, KBjBj); KBjBj.diag() += SQR(od->noise);
            KBjBj = inv_sympd(KBjBj);
            cov_matrices(i, j) = new mat(od->bSize, od->bSize);
            NFOR(u, v, od->bSize, od->bSize)
                cov_matrices(i, j)->at(u, v) = KBjBj(u, v);
            XBj.clear(); KBjBj.clear();
        }

        alpha.clear(); theta.clear(); s.clear();
    }

    double rmse = 0.0;
    int pos = 0;

    SFOR(i, od->nBlock)
    {
        mat XTi = (*od->getxt(i)), YTi = (*od->getyt(i)) - od->y_mean;
        SFOR(j, od->tSize)
        {
            rowvec xt = XTi.row(j);
            cout << "Predicting testing no. " << ++pos << " ..." << endl;
            rmse += fast_predict(M, b, xt, YTi.at(j, 0), cov_matrices); xt.clear();
        }
        XTi.clear(); YTi.clear();
    }

    return sqrt(rmse / (od->nBlock * od->tSize));
}

double predictor::fast_predict(mat &M, vec &b, rowvec &xt, double yt, bmat &cov_matrices)
{
    double max_var = 0.0, rmse = INFTY;
    vec variance(od->nBlock), pred(od->nBlock); variance.fill(0.0); pred.fill(0.0);

    SFOR(i, od->nBlock)
    {
        mat XBi = (*od->getxb(i)), YBi = (*od->getyb(i)) - od->y_mean, KtBi, W;

        SFOR(j, nz)
        {
            //cout << i << " " << j << endl;
            vec alpha = M * Z.row(j).t() + b;
            mat theta; vec s;
            unvectorise(theta, s, alpha, od->nDim, config->nBasis);
            bs->kernel(xt, XBi, theta, od->signal, KtBi);

            W = *cov_matrices(j, i); W = KtBi * W;
            pred(i) += (1.0 / nz) * trace(W * YBi);
            variance(i) += (1.0 / nz) * fabs(trace(W * KtBi.t()));
        }
    }

    SFOR(i, od->nBlock)
    if (max_var < variance(i))
    {
        max_var = variance(i);
        rmse = SQR(yt - pred(i));
    }

    return rmse;
}

double predictor::pro_predict(mat &M, vec &b, rowvec &xt, double yt)
{
    double min_var = INFTY, rmse = INFTY;
    vec variance(od->nBlock), pred(od->nBlock); variance.fill(0.0); pred.fill(0.0);

    SFOR(i, nz)
    {
        vec alpha = M * Z.row(i).t() + b;
        mat theta; vec s;
        unvectorise(theta, s, alpha, od->nDim, config->nBasis);

        SFOR(j, od->nBlock)
        {
            mat XBj = (*od->getxb(j)), YBj = (*od->getyb(j)) - od->y_mean;
            mat KBjBj, KtBj;

            bs->kernel(XBj, theta, od->signal, KBjBj);
            bs->kernel(xt, XBj, theta, od->signal, KtBj);

            KBjBj.diag() += SQR(od->noise);
            mat L = KtBj * inv_sympd(KBjBj);
            pred(j) += (1.0 / nz) * trace(L * YBj);
            variance(j) += (1.0 / nz) * fabs(bs->kernel(xt, theta, od->signal) - trace(L * KtBj.t()));

            XBj.clear(); YBj.clear(); KBjBj.clear(); KtBj.clear();
        }

        theta.clear(); s.clear();
    }

    SFOR(i, od->nBlock)
    if (variance(i) < min_var)
    {
        min_var = variance(i);
        rmse = SQR(pred(i) - yt);
    }

    return rmse;
}

double predictor::pro_predict(mat &M, vec &b)
{
    double rmse = 0.0;
    int pos = 0;

    SFOR(i, od->nBlock)
    {
        mat XTi = (*od->getxt(i)), YTi = (*od->getyt(i)) - od->y_mean;
        SFOR(j, od->tSize)
        {
            rowvec xt = XTi.row(j);
            cout << "Predicting testing no. " << ++pos << " ..." << endl;
            rmse += pro_predict(M, b, xt, YTi.at(j, 0)); xt.clear();
        }
        XTi.clear(); YTi.clear();
    }

    rmse /= (od->nBlock * od->tSize);

    return sqrt(rmse);
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

	//#pragma omp parallel for schedule(static, chunk)
    SFOR(i, nz)
    {
        int t = 0; //omp_get_thread_num();
        vec alpha = M * Z.row(i).t() + b;
        mat theta; vec s;
        unvectorise(theta, s, alpha, od->nDim, config->nBasis);

        //basis *localbs = new basis();

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

            KBjBj.clear(); KTjBj.clear(); pred.clear();
            XBj.clear(); YBj.clear(); XTj.clear(); YTj.clear();
        }

        theta.clear(); alpha.clear(); s.clear();
    }

    NFOR(i, j, od->nBlock, od->tSize)
    {
        double sum = 0.0; // hmm ...
        SFOR(t, nThread) sum += diff[t][i][j];
        rmse += SQR(sum);
    }

    return sqrt((1.0 / od->nTest) * rmse);
}

double predictor::combined_predict(mat &M, vec &b)
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

	//#pragma omp parallel for schedule(static, chunk)
    SFOR(i, nz)
    {
        int t = 0; //omp_get_thread_num();
        vec alpha = M * Z.row(i).t() + b;
        mat theta; vec s;
        unvectorise(theta, s, alpha, od->nDim, config->nBasis);

        SFOR(j, od->nBlock)
        {
            mat PhiTjs(od->tSize, 1);
            mat KBjBj, KTjBj;
            mat XBj = (*od->getxb(j));
            mat XTj = (*od->getxt(j));
            mat YBj = (*od->getyb(j)) - od->y_mean;
            mat YTj = (*od->getyt(j)) - od->y_mean;

            SFOR(k, od->tSize)
            {
                rowvec XTjk = XTj.row(k);
                colvec phi;
                bs->Phi(XTjk, phi, theta);
                PhiTjs(k, 0) = dot(phi, s);
                XTjk.clear(); phi.clear();
            }

            bs->kernel(XBj, theta, od->signal, KBjBj);
            bs->kernel(XTj, XBj, theta, od->signal, KTjBj);

            KBjBj.diag() += SQR(od->noise);

            mat pred = KTjBj * inv_sympd(KBjBj) * YBj + PhiTjs;
            diff[t][j] += (1.0 / nz) * (YTj - pred);

            KBjBj.clear(); KTjBj.clear(); pred.clear();
            XBj.clear(); YBj.clear(); XTj.clear(); YTj.clear();
        }

        theta.clear(); alpha.clear(); s.clear();
    }

    NFOR(i, j, od->nBlock, od->tSize)
    {
        double sum = 0.0; // hmm ...
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
