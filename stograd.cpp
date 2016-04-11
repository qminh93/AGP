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

void stodgrad::compute_dalpha_dM(vec &z)
{
    SFOR(k, z.n_rows)
    {
        dalpha[k] = zeros(z.n_rows, z.n_rows);
        dalpha[k].row(k) = z.t();
    }
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
        vec alpha = M * z + b;

        // Construct dalpha/dM
        dalpha = vm(alpha.n_rows);
        compute_dalpha_dM(z);

        // Extract from alpha
        mat theta;
        vec s;
        unvectorise(theta, s, alpha, od->nDim, nBasis)

        SFOR(j, nk)
        {
            vec Fkz, dlogqp;
            compute_F(theta, s, K(j), Fkz);
            compute_dlogqp(theta, s, K(j));
            res += od->nBlock * Fkz - dlogqp;
        }

        dalpha.clear(); theta.clear(); s.clear();
    }
    res = (1.0 / (nz * nk)) * res;
}

void stograd::compute_dlogqp(mat &theta, mat &M, vec &s, int k)
{
    int alpha_size = theta.n_rows * theta.n_cols + s.n_rows;
    dlogqp = - inv(M).t();

    int count = 0;

    NFOR(tr, tc, theta.n_rows, theta.n_cols) dlogqp -= theta(tc, tr) * dalpha[count++];
    SFOR(i, s.n_rows) dlogqp -= ((double)nBasis / (double)od->bSize) * s(i) * dalpha[count++];
}

void stograd::compute_F(mat &theta, vec &s, int k, vec &Fkz)
{
    vec vk, dvk;

    compute_vk(k, theta, s, vk);
    compute_dvk(k, theta, s, dvk);
    compute_rk();

    Fkz = -0.5 * rk;
}

void stograd::compute_vk(int k, mat &theta, vec &s)
{
    vk = vec(od->bSize); vec phi;
    mat* xk = od->getxb(k);
    mat* yk = od->getyb(k);

    SFOR(i, od->bSize)
    {
        rowvec xki = xk->row(i);
        basis->Phi(xki, phi, theta);
        vk(i) = yk->at(i, 1) - dot(phi, s);
    }

    phi.clear();
}

void stograd::compute_dvk(int k, mat &theta, vec &s)
{
    alpha_size = nBasis * od->nDim + s.n_rows;
    dvk        = vm (2 * od->bSize);

    SFOR(i, od->bSize)
    {
        dvk[i] = zeros<mat> (alpha_size, alpha_size);
        dvk[i + od->bSize] = zeros<mat> (alpha_size, 1);
    }

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
            dvk[i + od->bSize](j * od->nDim + t, 0) = dphi(t, 2 * j) * s(2 * j) + dphi(t, 2 * j + 1) * s(2 * j + 1);
        SFOR(j, s.n_rows)
            dvk[i + od->bSize](nBasis * od->nDim + j, 0) = phi(j);
        xki.clear();
        dvk[i + od->bSize] *= -1;
    }

    NFOR(i, j, od->bSize, alpha_size)
        dvk[i] += dvk[i + od->bSize](j, 0) * dalpha(j);
}

void stograd::compute_rk()
{
    int eta_size   = SQR(dvk[0].n_rows) + dvk[0].n_rows,
        alpha_size = dvk[0].n_rows;

    rk = vec(eta_size, 1);

    SFOR(i, od->bSize)
    {
        SFOR(Mr, Mc, alpha_size, alpha_size)
            rk[Mc * alpha_size + Mr] += dvk[i](Mr, Mc) * vk(i, 0);
        SFOR(Mr, alpha_size)
            rk[SQR(alpha_size) + Mr] += dvk[i + od->bSize](Mr, 0) * vk(i, 0);
    }

    rk *= 2.0 / SQR(od->noise);
}
