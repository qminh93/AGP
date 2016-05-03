#include "optimizer.h"

optimizer::optimizer(OrganizedData *od, configuration * config, int nBasis)
{
    this->od = od;
    this->nBasis = nBasis;
    this->config = config;
    this->gradient = new stograd(this->od, nBasis);
    this->savedItems.load("dump.bin");
}

optimizer::~optimizer()
{
    //dtor
}

pdd optimizer::learning_rate(int nIter) // compute the adaptive learning rate which decays as the number of iterations increases
{
    pdd rates;
    this->config->M_tau += 0.1 * sqrt(nIter);
    this->config->b_tau += 0.1 * sqrt(nIter);
    this->config->M_lambda += 0.1 * sqrt(nIter);
    this->config->b_lambda += 0.1 * sqrt(nIter);
    rates.first = this->config->M_r / pow(1 + nIter * this->config->M_tau * this->config->M_r, this->config->M_decay);
    rates.second = this->config->b_r / pow(1 + nIter * this->config->b_tau * this->config->b_r, this->config->b_decay);
    return rates;
}

void optimizer::initialize() // initialize M = <constant> * I & b = 0
{
    int m = this->nBasis, d = this->od->nDim;
    mat temp = randu <mat> (m * (d + 2), m * (d + 2));
    //this->M = savedItems(30, 0);
    //this->b = savedItems(30, 1);
    this->M = temp * temp.t() + eye <mat> (m * (d + 2), m * (d + 2)); this->dM = 0.0 * this->M;
    this->b = randu <vec> (m * (d + 2)); this->db = this->b;
    vectorise(M, b, this->eta); // eta = vec(M, b)
    this->z_mean = zeros <vec> (m * (d + 2)); // psi(z) = N(0, I) -- later, we parameterise alpha | y = Mz + b which implies q(alpha) = (1/|M|) * psi(z)
    this->z_cov = eye <mat> (m * (d + 2), m * (d + 2));
    this->model = new predictor(DEFAULT_NO_SAMPLE, this->z_mean, this->z_cov, this->od, this->config); // initialize a predictor
}

void optimizer::optimize()
{
    cout << "Initializing parameters ..." << endl;
    t1 = time(NULL);
    initialize();
    savedItems(0, 0) = M;
    savedItems(0, 1) = b;
    t2 = time(NULL);
    //cout << "Incurred time = " << t2 - t1 << " secs." << endl;
    int Ms = this->nBasis * (this->od->nDim + 2); // evaluate |z| which is also |alpha|

    cout << "Evaluating performance before learning ..." << endl;
    t1 = time(NULL);
    vector < pair <int, double> > recorder;
    pair <int, double> p; p.first = 0; p.second = this->model->fast_predict(this->M, this->b);
    recorder.push_back(p);
    t2 = time(NULL);

    cout << "RMSE = " << p.second << endl;
    //cout << "Incurred time = " << t2 - t1 << " secs." << endl;
    cout << "Start updating ..." << endl;
    SFOR(t, this->config->training_num_ite)
    {
        cout << "Iteration " << t + 1 << " ..." << endl;
        cout << "Sampling ..." << endl;
        t1 = time(NULL);
        mat Z(this->config->anytime_z_sample, Ms);
        vec K(this->config->anytime_k_sample), stgrad(this->eta.size());
        multi_gaussian_random(this->z_mean, this->z_cov, this->config->anytime_z_sample, Z); // draw anytime_z_sample samples of Z from psi(z); one sample per row
        SFOR(i, this->config->anytime_k_sample) K(i) = IRAND(0, this->od->nBlock - 1); // draw anytime_k_sample of k from U(0, p - 1)
        t2 = time(NULL);
        //cout << "Incurred time = " << t2 - t1 << " secs." << endl;
        cout << "Computing stochastic gradient ..." << endl;
        t1 = time(NULL);
        mat Zt = Z.t(); // each sample is now stored as a column
        this->gradient->compute(Zt, K, this->eta, Ms, Ms, stgrad); // compute the corresponding stochastic gradient
        t2 = time(NULL);
        //cout << "Incurred time = " << t2 - t1 << " secs." << endl;
        cout << "Extracting results ..." << endl;
        t1 = time(NULL);
        stgrad = stgrad / norm(stgrad); // normalize the gradient since we only care about its direction
        unvectorise(this->dM, this->db, stgrad, Ms, Ms); // extracting the corresponding gradient with respect to M and b
        t2 = time(NULL);
        //cout << "Incurred time = " << t2 - t1 << " secs." << endl;
        cout << "Computing adaptive learning rates ..." << endl;
        t1 = time(NULL);
        pdd rates = learning_rate(t); // compute the adaptive learning rate
        t2 - time(NULL);
        // cout << "Incurred time = " << t2 - t1 << " secs." << endl;
        cout << "Updating ..." << endl;
        t1 = time(NULL);
        this->M += rates.first * (this->dM - this->config->M_lambda * this->M); // update with regularisation
        this->b += rates.second * (this->db - this->config->b_lambda * this->b); // update with regularisation
        savedItems(t + 1, 0) = M;
        savedItems(t + 1, 1) = b;
        vectorise(this->M, this->b, this->eta); // combine M & b into eta again
        t2 = time(NULL);
        //cout << "Incurred time = " << t2 - t1 << " secs." << endl;
        if (((t + 1) % this->config->anytime_interval) == 0) // printing out the RMSE periodically ...
        {
            cout << "Predicting ..." << endl;
            t1 = time(NULL);
            p.first = t + 1; p.second = this->model->fast_predict(this->M, this->b);
            cout << "RMSE = " << p.second << endl;
            recorder.push_back(p);
            t2 = time(NULL);
           // cout << "Incurred time = " << t2 - t1 << " secs." << endl;
        }
    }
    cout << "Done. To summarize: " << endl;
    SFOR(t, recorder.size()) cout << recorder[t].first << " " << recorder[t].second << endl;
    savedItems.save("dump.bin");
}
