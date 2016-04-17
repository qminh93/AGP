#include "optimizer.h"

optimizer::optimizer(OrganizedData *od, configuration * config, int nBasis)
{
    this->od = od;
    this->nBasis = nBasis;
    this->config = config;
    this->gradient = new stograd(this->od, nBasis);
}

optimizer::~optimizer()
{
    //dtor
}

pdd optimizer::learning_rate(int nIter) // compute the adaptive learning rate which decays as the number of iterations increases
{
    pdd rates;
    rates.first = this->config->M_r / pow(1 + nIter * this->config->M_tau * this->config->M_r, this->config->M_decay);
    rates.second = this->config->b_r / pow(1 + nIter * this->config->b_tau * this->config->b_r, this->config->b_decay);
    return rates;
}

void optimizer::initialize() // initialize M = <constant> * I & b = 0
{
    int m = this->nBasis, d = this->od->nDim;
    this->M = eye <mat> (m * (d + 2), m * (d + 2)); this->dM = 0.0 * this->M;
    this->b = zeros <vec> (m * (d + 2)); this->db = this->b;
    vectorise(M, b, this->eta); // eta = vec(M, b)
    this->z_mean = zeros <vec> (m * (d + 2)); // psi(z) = N(0, I) -- later, we parameterise alpha | y = Mz + b which implies q(alpha) = (1/|M|) * psi(z)
    this->z_cov = eye <mat> (m * (d + 2), m * (d + 2));
    this->model = new predictor(DEFAULT_NO_SAMPLE, this->z_mean, this->z_cov, this->od, this->config); // initialize a predictor
}

void optimizer::optimize()
{
    cout << "Initializing parameters ..." << endl;
    initialize();
    int Ms = this->nBasis * (this->od->nDim + 2); // evaluate |z| which is also |alpha|

    cout << "Evaluating performance before learning ..." << endl;
    cout << "RMSE = " << this->model->predict(this->M, this->b) << endl;

    cout << "Start updating ..." << endl;
    SFOR(t, this->config->training_num_ite)
    {
        cout << "Iteration " << t + 1 << " ..." << endl;
        cout << "Sampling ..." << endl;
        mat Z(this->config->anytime_z_sample, Ms);
        vec K(this->config->anytime_k_sample), stgrad(this->eta.size());
        multi_gaussian_random(this->z_mean, this->z_cov, this->config->anytime_z_sample, Z); // draw anytime_z_sample samples of Z from psi(z); one sample per row
        SFOR(i, this->config->anytime_k_sample) K(i) = IRAND(0, this->od->nBlock - 1); // draw anytime_k_sample of k from U(0, p - 1)

        cout << "Computing stochastic gradient ..." << endl;
        mat Zt = Z.t(); // each sample is now stored as a column
        this->gradient->compute(Zt, K, this->eta, Ms, Ms, stgrad); // compute the corresponding stochastic gradient

        cout << "Extracting results ..." << endl;
        stgrad = stgrad / norm(stgrad); // normalize the gradient since we only care about its direction
        unvectorise(this->dM, this->db, stgrad, Ms, Ms); // extracting the corresponding gradient with respect to M and b

        cout << "Computing adaptive learning rates ..." << endl;
        pdd rates = learning_rate(t); // compute the adaptive learning rate

        cout << "Updating ..." << endl;
        this->M += (rates.first - this->config->M_lambda) * this->dM; // update with regularisation
        this->b += (rates.second - this->config->b_lambda) * this->db; // update with regularisation
        vectorise(this->M, this->b, this->eta); // combine M & b into eta again

        if (((t + 1) % this->config->anytime_interval) == 0) // printing out the RMSE periodically ...
        {
            cout << "Predicting ..." << endl;
            cout << "RMSE = " << this->model->predict(this->M, this->b) << endl;
        }
    }
    cout << "Done." << endl;
}
