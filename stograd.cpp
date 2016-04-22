#include "stograd.h"
#define DEBUG 0

stograd::stograd(OrganizedData* od, int nBasis)
{
    this->od     = od; // store the pointer to the organizing module
    this->nBasis = nBasis; // the number of basis function, i.e., m -- in fact, we have 2m basis functions but they go in pair of (cos(.), sin(.)) so we treat each pair as one basis function
    this->bs     = new basis(); // create and maintain a basis function object to assist the differentiating job
}

stograd::~stograd() // clear memory
{
    M.clear();
    b.clear();
}

void stograd::compute_dalpha_dM(vec &z, vm &dalpha) // compute {dalpha_k/dM}_k -- remember that alpha = Mz + b where M & b are parameters
// hence, dalpha_k/dM = [dalpha_k/dM_{ij}]_{ij} = {I(i = k)zj}_{ij} -- this is because alpha_k = \sum_j M_{kj}zj
{
    SFOR(k, z.n_rows) // for each k
    {
        dalpha[k] = zeros(z.n_rows, z.n_rows); // initialise dalpha_k/dM
        dalpha[k].row(k) = z.t(); // set values for its k^th row, i.e., {I(i = k)zj}_k = z' if k = i. Otherwise, it is just zero;
    }
}

void stograd::compute(mat &Z, vec &K, vec &eta, int M_r, int M_c, vec &res) // compute a stochastic version of dL/deta
// following the paper, it is 1 / (nz * nk) \sum_{kz} (pFkz(eta, alpha_z) - (d/deta)(log(q(alpha_z)/p(alpha_z))))
// where eta = vec(M, b), p is the number of data blocks and alpha_z = Mz + b
// nz is the number of z-samples drawn independently from N(0, I)
// nk is the number of block indices drawn independently from U(1, p) or U(0, p - 1) -- the latter is the C++ 0-based indexing
{
    unvectorise(M, b, eta, M_r, M_c); // extract M and b from eta
    M_inv = - inv(M).t();

    res = zeros <vec> (eta.n_rows);
    int nz = Z.n_cols, nk = K.n_rows;

    int nThread = omp_get_num_procs(), chunk = nz / nThread;
	if (chunk == 0) chunk++;

	vm res_t(nThread); SFOR(i, nThread) res_t[i] = vec(eta.n_rows);

	#pragma omp parallel for schedule(dynamic, chunk)
    SFOR(i, nz) // for each z-sample
    {
        int t = omp_get_thread_num();
        // construc alpha_z
        vec z = Z.col(i);
        vec alpha = M * z + b;

        // construct dalpha/dM
        vm dalpha(alpha.n_rows);
        compute_dalpha_dM(z, dalpha);

        // extract theta and s from alpha
        mat theta; vec s; // theta is a d x m matrix and s is a (2 * m x 1) column vector
        unvectorise(theta, s, alpha, od->nDim, nBasis); // alpha is a m * (d + 2) x 1 column vector

        vec dlogqp;
        compute_dlogqp(theta, s, dalpha, dlogqp); // compute (d/deta)(log(q(alpha_z)/p(alpha_z)))

        SFOR(j, nk) // for each sampled block index
        {
            vec Fkz; // Fkz is supposed to be a ((size(alpha)^2 + size(alpha)) x 1) column vector
            compute_F(theta, s, K(j), dalpha, Fkz); // compute Fkz(eta, alpha_z)
            //compute_dlogqp(theta, s, K(j)); // compute (d/deta)(log(q(alpha_z)/p(alpha_z)))
            res_t[t] += (od->nBlock * Fkz - dlogqp); // updating dL/deta
            Fkz.clear();
        }

        dalpha.clear(); theta.clear(); s.clear(); z.clear(); alpha.clear(); // clear memory
    }

    SFOR(t, nThread) res += res_t[t];
    res = (1.0 / (nz * nk)) * res; // return dL/deta
}

void stograd::compute_dlogqp(mat &theta, vec &s, vm &dalpha, vec &dlogqp) // compute (d/deta)(log(q(alpha)/p(alpha))) given alpha
// this is achieved by computing (d/dM)(log(q(alpha)/p(alpha))) and (d/db)(log(q(alpha)/p(alpha)))
// (d/deta) can then be constructed as vec((d/dM), (d/db))
{
    mat dlogqp_dM = M_inv; // note that (d/dM)(logqp) = -(M^{-1})' - sum_k (dalpha_k/dM * (d/dalpha_k)(log p(alpha)))
    vec dlogqp_db(M.n_rows);
    // note that (d/db) (logqp) = (d/dalpha) log(p(alpha)) = -blkdiag[I_md, (m / signal^2) * I_2m] * alpha
    // which simplifies as (d/db) (logqp) = [-vec(theta); - (m / signal^2) * s since alpha = vec(theta, s)]
    int cc = 0; // cc will iterate through all component indices of alpha

    NFOR(tc, tr, theta.n_cols, theta.n_rows) // for each theta_tc and each component tr of the column vector theta_tc
    {
        dlogqp_db[cc] = -theta(tr, tc); // this follows from the above simplification of (d/db) -- bug: wrong sign (fixed)
        dlogqp_dM += theta(tr, tc) * dalpha[cc++]; // wrong sign since dlogqp_db[cc] = -theta(tr, tc) (fixed)
    }

    SFOR(i, s.n_rows) // for each component of the column vector s
    {
        dlogqp_db[cc] = -((double)nBasis / SQR(od->signal)) * s(i); // this follows from the simplified expression for (d/db)
        // potential bug: shouldn't it be nBasis / signal^2 instead of nBasis / bSize (fixed)
        // potential bug: wrong sign (fixed)
        dlogqp_dM += ((double)nBasis / SQR(od->signal)) * s(i) * dalpha[cc++];
        // potential bug: shouldn't it be nBasis / signal^2 instead of nBasis / bSize (fixed)
        // potential bug: wrong sign (fixed)
    }

    vectorise(dlogqp_dM, dlogqp_db, dlogqp); // (d/deta) = vec((d/dM), (d/db))
}

void stograd::compute_F(mat &theta, vec &s, int k, vm &dalpha, vec &Fkz) // compute Fkz = -0.5 * rk where
// rk = [rk(u)]' with rk(u) = (1 / noise^2) * (d/du) (vk' * vk) = (2 / noise^2) * (dvk/du)' * vk & u iterates through the component of eta
{
    vec vk, rk;
    vm dvk;

    compute_vk(k, theta, s, vk); // compute vk
    compute_dvk(k, theta, s, dalpha, dvk); // compute dvk
    compute_rk(vk, dvk, rk); // compute rk = [rk(u)]' with u iterates through the component of eta & rk(u) = (2 / noise^2) * (dvk/du)' * vk
    Fkz = -0.5 * rk; // Fkz = -0.5 * rk
}

void stograd::compute_vk(int k, mat &theta, vec &s, vec &vk) // given theta and s, compute vk = yk - Phi(Xk)' * s for data block k^th
{
    colvec phi;

    vk = vec(od->bSize); // vk is a bSize x 1 column vector where bSize is the size of one block of data
    mat* xk = od->getxb(k); // the inputs of block k
    mat* yk = od->getyb(k); // the noisy outputs of block k

    SFOR(i, od->bSize) // for each input in block k
    {
        rowvec xki = xk->row(i);
        bs->Phi(xki, phi, theta); // compute the feature vector phi = Phi(xki) given theta
        vk(i) = (yk->at(i, 0) - od->y_mean) - dot(phi, s); // vki = yki - Phi(xki)' * s -- we consider y to be normalized
        // potential bug: shouldn't it be yk->at(i, 0) instead of yk->at(i, 1) ? (fixed)
    }

    phi.clear(); // clear memory
}

void stograd::compute_dvk(int k, mat &theta, vec &s, vm &dalpha, vm &dvk) // compute dvk/deta = [dvk/du]' with u iterates through the components of eta
// this is achieved by computing dvki/du for every pair of (i, u) s.t i \in block k & u \in alpha first
// then, using chain rule of derivative to construct dvki/du for u in eta
{
    int alpha_size = nBasis * od->nDim + s.n_rows; // compute the size of alpha = vec(theta, s)
    dvk = vm (2 * od->bSize); // for each i \in block k, we need to store the result of dvki/du for u \in M and dvki/du for u \in b
    // each of which is stored in a matrix so we need 2 * |block k| = 2 * bSize matrices

    SFOR(i, od->bSize) // initialization
    {
        dvk[i] = zeros <mat> (alpha_size, alpha_size); // for the first half of dvk, we store dvki/du for u \in M, hence we need a matrix of size |alpha| by |alpha|
        dvk[i + od->bSize] = zeros <mat> (alpha_size, 1); // for the second half of dvk, we only need a vector to store dvki/du for u \in b since b is a vector of size |alpha| by 1
    }

    mat dphi; vec phi;
    mat* xk = od->getxb(k); // extract the input of block k

    SFOR(i, od->bSize) // for each input xki in block k, let's compute dvki/dalpha
    {
        rowvec xki = xk->row(i); // extract xki
        bs->Phi(xki, phi, theta); // compute phi = Phi(xki)
        bs->dPhi_dtheta(xki, theta, phi, dphi); // compute dphi = dPhi(xki) = [dcos(2 * pi * xki * theta_1)/dtheta_1 dsin(2 * pi * xki * theta_1)/dtheta_1 ... dcos(2 * pi * xki * theta_m)/dtheta_m dsin(2 * pi * xki * theta_m)/dtheta_m]
        NFOR(j, t, nBasis, od->nDim) // for each variable ujt in theta (indexed by (j, t))
            dvk[i + od->bSize](j * od->nDim + t, 0) = dphi(t, 2 * j) * s(2 * j) + dphi(t, 2 * j + 1) * s(2 * j + 1); // compute dvki/dujt where ujt = theta_j(t)
            // dvki/dujt = -dcos(2 * pi * xki * theta_j)[t] * s(2 * j) - dsin(2 * pi * xki * theta_j)[t] * s(2 * j + 1)
            // since dcos(2 * pi * xki * theta_j) = dphi(., 2 * j) & dsin(2 * pi * xki * theta_j) = dphi(., 2 * j + 1), the above formulation follows; we will deal with the minus sign later
        SFOR(j, s.n_rows) // for each variable sj in s (indexed by j)
            dvk[i + od->bSize](nBasis * od->nDim + j, 0) = phi(j);
            // since dvki/ds = -d(Phi(xki)' * s)/ds = -Phi(xki) = -phi, dvki/dsj = -phi(j) -- we will deal with the minus sign later
        xki.clear(); // clear memory
        dvk[i + od->bSize] *= -1.0; // since dvki = dyki - d(Phi(xki)' * s) = -d(Phi(xki)' * s)
    }

    NFOR(i, j, od->bSize, alpha_size) // since dvki/db = (dalpha/db) * dvki/dalpha = I * dvki/dalpha = dvki/dalpha, we have already got dvki/db stored in the second half of dvk
        dvk[i] += dvk[i + od->bSize](j, 0) * dalpha[j]; // dvki/dM = sum_j (dvki/dalpha_j) * (dalpha_j/dM) by chain rule of derivatives
}

void stograd::compute_rk(vec &vk, vm &dvk, vec &rk) // compute rk = [rk(u)]' where u iterates through variables in eta = vec(M, b)
{
    int eta_size   = SQR(dvk[0].n_rows) + dvk[0].n_rows,
        alpha_size = dvk[0].n_rows; // recall that |eta| = |alpha|^2 + |alpha|

    rk = vec(eta_size, 1); // rk is a |eta| x 1 column vector

    SFOR(i, od->bSize) // iterating through each index component of vk
    {
        NFOR(Mr, Mc, alpha_size, alpha_size) // for u \in M, indexed by (Mr, Mc)
            rk[Mc * alpha_size + Mr] += dvk[i](Mr, Mc) * vk(i, 0); // compute rk[u] = \sum_i (dvki/du * vki)
        SFOR(Mr, alpha_size) // for u \in b, indexed by Mr
            rk[SQR(alpha_size) + Mr] += dvk[i + od->bSize](Mr, 0) * vk(i, 0); // compute rk[u] = \sum_i (dvki/du * vki)
    }

    rk *= 2.0 / SQR(od->noise); // remember rk[u] = (2.0 / noise^2) * (dvk/du)' * vk so we have to multiply the final result by (2.0 / noise^2)
}
