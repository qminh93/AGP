#include "optimizer.h"

optimizer::optimizer(OrganizedData *od, configuration * config, int nBasis)
{
    this->od = od;
    this->nBasis = nBasis;
    this->config = config;
    this->gradient = new stograd(this->od);
}

optimizer::~optimizer()
{
    //dtor
}

pdd optimizer::learning_rate(int nIter)
{
    pdd rates;
    rates.first = this->config->M_r / pow(1 + nIter * this->config->M_tau * this->config->M_r, this->config->M_decay);
    rates.second = this->config->b_r / pow(1 + nIter * this->config->b_tau * this->config->b_r, this->config->b_decay);
    return rates;
}

void optimizer::initialize()
{
    int m = this->nBasis, d = this->od->nDim;
    this->M = eye <mat> (m * (d + 2), m * (d + 2)); this->dM = this->M;
    this->b = zeros <vec> (m * (d + 2)); this->db = this->b;
    vectorise(M, b, this->eta);
    this->z_mean = zeros <vec> (m * (d + 2));
    this->z_cov = eye <mat> (m * (d + 2), m * (d + 2));
    this->model = new predictor(DEFAULT_NO_SAMPLE, this->z_mean, this->z_cov, this->od, this->config);
}

void optimizer::optimize()
{
    cout << "Initializing parameters ..." << endl;
    initialize();
    int Ms = this->nBasis * (this->od->nDim + 2);
    cout << "Start updating ..." << endl;
    SFOR(t, this->config->training_num_ite)
    {
        cout << "Iteration " << t + 1 << " ..." << endl;
        cout << "Sampling ..." << endl;
        mat Z(this->config->anytime_z_sample, Ms);
        vec K(this->config->anytime_k_sample), stgrad(this->eta.size());
        multi_gaussian_random(this->z_mean, this->z_cov, this->config->anytime_z_sample, Z);
        SFOR(i, this->config->anytime_k_sample) K(i) = IRAND(0, this->od->nBlock - 1);
        cout << "Computing stochastic gradient ..." << endl;
        this->gradient->compute(Z.t(), K, this->eta, Ms, Ms, stgrad);
        unvectorise(this->dM, this->db, stgrad, Ms, Ms);
        cout << "Computing adaptive learning rates ..." << endl;
        pdd rates = learning_rate(t);
        cout << "Updating ..." << endl;
        this->M += rates.first * this->dM;
        this->b += rates.second * this->db;
        vectorise(this->M, this->b, this->eta);

        if (((t + 1) % this->config->anytime_interval) == 0)
        {
            cout << "Predicting ..." << endl;
            cout << "RMSE = " << this->model->predict(this->M, this->b) << endl;
        }
    }
    cout << "Done." << endl;
}
