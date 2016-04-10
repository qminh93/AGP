#include "stograd.h"

stograd::stograd(OrganizedData* od, int nBasis)
{
    this->od     = od;
    this->nBasis = nBasis;
    this->bs     = new basis();
}

stograd::~stograd()
{
    M.clear();
    b.clear();
}

void stograd::compute(mat &Z, vec &K, vec &eta, int M_r, int M_c, vec &res)
{
    unvectorise(M, b, eta, M_r, M_c);

    res = vec(eta.n_rows, 0);
    int nz = Z.n_cols, nk = K.n_rows;
    SFOR(i, nz)
    {
        // Construc sample alpha
        vec z = Z.col(i);
        alpha = M * z + b;



        // Extract from alpha
        mat theta;
        vec s;
        unvectorise(theta, s, alpha, od->nDim, nBasis)

        SFOR(j, nk)
        {

            int k = K(j);
            vec Fkz, dlogqp;
            compute_F(theta, s, k, Fkz);
            compute_dlogqp(theta, s, k, dlogqp)
            res += od->nBlock * Fkz - dlogqp;
        }
    }
    res = (1.0 / (nz * nk)) * res;
}

void stograd::compute_F(mat &theta, vec &s, int k, vec &Fkz)
{
    vec vk, dvk;

    compute_vk(k, theta, s, vk);
    compute_dvk(k, theta, s, dvk);
}

void stograd::compute_vk(int k, mat &theta, vec &s, vec &vk)
{
    vk = vec(od->bSize);
    vec phi;
    mat* xk = od->getxb(k);
    mat* yk = od->getyb(k);

    SFOR(i, od->bSize)
    {
        rowvec xki = xk->row(i);
        basis->Phi(xki, phi, theta);
        vk(i) = yk->at(i, 1) - dot(phi, s);
    }
}

void stograd::compute_dvk_dalpha(int k, mat &theta, vec &s, mat &dvk)
{
    alpha_size = nBasis * od->nDim + s.n_rows;
    dvk = mat(alpha_size, od->bSize);
    mat dphi;
    vec phi;
    mat* xk = od->getxb(k);
    mat* yk = od->getyb(k);

    SFOR(i, od->bSize)
    {
        rowvec xki = xk->row(i);
        basis->Phi(xki, phi, theta);
        basis->dPhi_dtheta(xki, theta, phi, dphi);
        NFOR(j, t, nBasis, od->nDim)
            dvk(j * od->nDim + t, i) = dphi(t, 2 * j) * s(2 * j) + dphi(t, 2 * j + 1) * s(2 * j + 1);
        SFOR(j, s.n_rows)
            dvk(nBasis * od->nDim + j, i) = phi(j);
        xki.clear();
    }
    dvk = - dvk;
}
