#include "basis.h"


basis::basis()
{
    //ctor
}

basis::~basis()
{
    //dtor
}

void basis::Phi(rowvec &x, colvec &phi, mat &theta) // theta = [theta_1 ... theta_m]; each theta_i is a column vector
{
    phi = colvec(2 * theta.n_cols); // phi = Phi(x) is a 2m x 1 column vector
    SFOR(i, theta.n_cols)
    {
        double dp       = dot(theta.col(i), x); // dp = x * theta_i
        phi(2 * i)      = cos(2 * PI * dp); // cos(2 * PI * x * theta_i)
        phi(2 * i + 1)  = sin(2 * PI * dp); // sin(2 * PI * x * theta_i)
    }
}

void basis::dPhi_dtheta(rowvec &x, mat &theta, colvec &phi, mat &dphi) // compute dphi(x) = [dcos(2 * pi * x * theta_1)/dtheta_1 dsin(2 * pi * x * theta_1)/dtheta_1 ... dcos(2 * pi * x * theta_m)/dtheta_m dsin(2 * pi * x * theta_m)/dtheta_m]
{
    dphi = zeros <mat> (x.n_elem, 2 * theta.n_cols); // dphi is a d x 2m matrix -- this can be easily verified from the above definition of dphi(x)

    SFOR(i, theta.n_cols) // for each component of phi(x), i.e., cos(2 * pi * x * theta_i) and sin(2 * pi * x * theta_i) for every 0 <= i <= m - 1
    {
        dphi.col(2 * i)     = - 2 * PI * phi(2 * i + 1) * x.t(); // compute dcos(2 * pi * x * theta_i)/dtheta_i = -2 * pi * sin(2 * pi * x * theta_i) * x' = -2 * pi * phi(2 * i + 1) * x'
        dphi.col(2 * i + 1) = 2   * PI * phi(2 * i)     * x.t(); // compute dsin(2 * pi * x * theta_i)/dtheta_i = 2 * pi * cos(2 * pi * x * theta_i) * x' = 2 * pi * phi(2 * i) * x'
    }
}

void basis::dPhi_dtheta(rowvec &x, mat &theta, mat &dphi) // compute dphi(x) = [dcos(2 * pi * x * theta_1)/dtheta_1 dsin(2 * pi * x * theta_1)/dtheta_1 ... dcos(2 * pi * x * theta_m)/dtheta_m dsin(2 * pi * x * theta_m)/dtheta_m]
// in this case, phi = Phi(x) has not been precomputed so we have to compute it first
{
    vec phi; Phi(x, phi, theta); // computing phi = Phi(x)
    dPhi_dtheta(x, theta, phi, dphi); // then, invoke the above computing procedure with phi = Phi(x) precomputed
}

double basis::kernel(rowvec &a, rowvec &b, mat &theta, double signal)
{
    colvec phi_a, phi_b;
    Phi(a, phi_a, theta);
    Phi(b, phi_b, theta);
    return (SQR(signal) / theta.n_cols) * dot(phi_a,phi_b);
}

double basis::kernel(rowvec &a, mat &theta, double signal)
{
    return kernel(a, a, theta, signal);
}

void basis::kernel(mat &A, mat &B, mat &theta, double signal, mat &KAB)
{
    KAB = mat(A.n_rows, B.n_rows);

    SFOR(i, A.n_rows)
    {
        colvec phi_Ai;
        rowvec Ai = A.row(i);
        Phi(Ai, phi_Ai, theta);
        SFOR(j, B.n_rows)
        {
            rowvec Bj = B.row(j);
            colvec phi_Bj;
            Phi(Bj, phi_Bj, theta);
            KAB(i, j) = dot(phi_Ai,phi_Bj);
        }
    }

    KAB = (SQR(signal) / theta.n_cols) * KAB;
}

void basis::kernel(mat &A, mat &theta, double signal, mat &KAA)
{
    kernel(A, A, theta, signal, KAA);
}
