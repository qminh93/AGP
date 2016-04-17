#include "libhead.h"

void randsample(int popSize, int sampleSize, vi &result)
{
    vi temp(popSize);
    SFOR(i, popSize) temp[i] = i;
    random_shuffle(temp.begin(),temp.end());
    result.clear();
    result = vi(temp.begin(), temp.begin() + sampleSize);
}

void r2v (rowvec &R, vd &result)
{
    result.clear();
    SFOR(i, R.n_elem) result.push_back(R(i));
}

void c2v (colvec &C, vd &result)
{
    result.clear();
    SFOR(i, C.n_elem) result.push_back(C(i));
}

void v2m (vvd &A, mat &result)
{
    result = mat (A.size(), A[0].size());
    if (A.size() == 0) return;
    NFOR(i, j, A.size(), A[0].size()) result(i,j) = A[i][j];
}

void shuffle(char* original, char* sample, int nTotal, int nSample)
{
	vector <int> pool(nTotal, 0);
	vector <int> mark(nTotal, 0);
	SFOR(i, nTotal) pool[i] = i;

	int nMax = nTotal - 1;
	SFOR(i, nSample)
	{
		int pos = IRAND(0, nMax), s = pool[pos];
		pool[pos] = pool[nMax]; nMax--;
		mark[s] = 1;
	}

	ifstream cin(original);
	ofstream cout(sample);

	string line;
	SFOR(i, nTotal)
	{
		cin >> line;
		if (mark[i] == 1) cout << line << endl;
	}

	cin.close(); cout.close();
}

double grand() // Marsaglia polar method -- generate a sample from N(0, 1)
{
    double u, v, s = 2.0;
    while ((s >= 1.0) || (s == 0.0))
    {
        u = DRAND(-1, 1);
        v = DRAND(-1, 1);
        s = u * u + v * v;
    }
    return u * sqrt(-2 * log(s) / s);
}

void multi_gaussian_random(vec &mean, mat &cov, vec &x) // generate a sample from N(mean, cov)
{
    mat L = chol(cov, "lower");
    int N = (int) mean.n_elem;

    vec z = zeros <vec> (N); z.fill(0.0);
    for (int i = 0; i < N; i++) z(i) = GRAND(0, 1);

    x = zeros <vec> (N); x.fill(0.0);
    for (int i = 0; i < N; i++)
    {
        x(i)= mean(i);
        for (int j = 0; j < N; j++) x(i) += (L(i, j) * z(j));
    }
}

void multi_gaussian_random(vec &mean, mat &cov, int N, mat &res) // generate N samples independently from N(mean, cov)
{
    res = mat(N, mean.n_elem); // one sample per row
    for (int i = 0; i < N; i++)
    {
        vec sample(mean.n_elem);
        multi_gaussian_random(mean, cov, sample);
        for (int j = 0; j < mean.n_elem; j++) res(i, j) = sample(j);
    }
}

void grand(double mean, double sigma, int N, vec &x) // generate N samples independently from N(mean, sigma^2)
{
    x = zeros <vec> (N);
    for (int i = 0; i < N; i++)
        x(i) = GRAND(mean, sigma);
}

void vectorise(mat &theta, vec &s, vec &res)
{
    int t_r   = theta.n_rows;
    int t_c   = theta.n_cols;
    int s_r   = s.n_rows;
    int index = 0;

    res = vec(s_r + t_r * t_c);

    NFOR(i, j, t_c, t_r) res(index++) = theta(j, i);
    SFOR(i, s_r) res(index++) = s(i);
}

void unvectorise(mat &theta, vec &s, vec &res, int t_r, int t_c)
{
    int s_r   = res.n_rows - t_r * t_c;
    int index = 0;

    theta = mat(t_r, t_c);
    s = vec(s_r);

    NFOR(i, j, t_c, t_r) theta(j, i) = res(index++);
    SFOR(i, s_r) s(i) = res(index++);
}


