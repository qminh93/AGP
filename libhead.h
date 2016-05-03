#ifndef LIBHEAD_H
#define LIBHEAD_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <armadillo>
#include <algorithm>
#include <fstream>
#include <vector>
#include <queue>
#include <deque>
#include <set>
#include <map>
#include <cstring>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <stack>
#include <bitset>
#include <functional>
#include <numeric>

using namespace std;
using namespace arma;

//#define SEED_DEFAULT 4111987
//#define SEED_DEFAULT 20051987
#define SEED_DEFAULT 26031993
//#define SEED_DEFAULT 8071950
//#define SEED_DEFAULT 10081950

/* UTILITY FUNCTIONS */
#define SEED(x) srand(x)
#define IRAND(a, b) (rand() % ((b) - (a) + 1) + (a)) // randomly generate an integer from [a, b]
#define DRAND(a, b) (((b) - (a)) * ((double) rand() / (RAND_MAX)) + (a)) // randomly generate a real number from [a, b]
#define EPS 0.00000000000000001
#define PI 3.141592653589793238462
#define EN 2.71828182845904509080
#define INFTY 1000000000000000.0
#define N(m, mean, var) ((1 / sqrt(2 * PI * (var))) * exp(-0.5 * (pow((m) - (mean), 2.0) / (var)))) // the pdf function for N(mean, var = sigma^2)
#define ENT(var) (0.5 * log(2.0 * PI * E * (var))) // the entropy of N(., var)
#define LN(m, mean, var) (-0.5 * log(2 * PI * (var)) - 0.5 * (pow((m) - (mean), 2.0)) / (var) ) // the log pdf function for N(mean, var = sigma^2)
#define SFOR(i,n) for (int i = 0; i < (int)n; i++)
#define NFOR(i,j,n,m) SFOR(i,n) SFOR(j,m)
#define CHUNK(a,b) (a > b ? a / b : 1)
#define SQR(a) ((a) * (a))
#define NUM2STR(x) ( static_cast < ostringstream* > (&(ostringstream() << x))->str() )
#define LAPSE(tstart, tend) ( 1000.0 * (tEnd - tStart) / CLOCKS_PER_SEC )
#define GRAND(mu, sigma) ((mu) + (sigma) * grand()) // randomly generate a real number from N(mu, sigma^2)

/* TYPE MACRO */
#define pdd	 pair 	<double, double>
#define vi   vector < int >
#define vd   vector < double >
#define vm   vector < mat >
#define vs   vector < string >
#define vvi  vector < vi >
#define vvd  vector < vd >
#define vvm  vector < vm >
#define ves  vector < ExpSetting* >
#define bmat field  < mat* >

struct Partition
{
	int nBlock; // the number of partitions
	vvd C; // store the estimated centroids
	vvi member; // lists of data indices belonging to each cluster
	vi  nAssign; // nAssign[i] -- the cluster which the ith data point belongs to

	Partition(int nBlock, vvd &C, vvi &member, vi &nAssign)
	{
		this->nBlock  = nBlock;
		this->C       = C;
		this->member  = member;
		this->nAssign = nAssign;
	}

	~Partition()
	{
		for (int i = 0; i < (int) C.size(); i++)
			C[i].clear();
		for (int i = 0; i < (int) member.size(); i++)
			member[i].clear();
		C.clear(); member.clear(); nAssign.clear();
	}
};

void randsample(int popSize, int sampleSize, vi &result); // randomly select a subset of size sampleSize from [0 ... popSize - 1] & store in vi
void r2v (rowvec &R, vd  &result); // ARMA rowvec to C++ vector
void c2v (colvec &C, vd  &result); // ARMA colvec to C++ vector
void v2m (vvd    &A, mat &result); // C++ vvd to ARMA mat structure
void shuffle(char* original, char* sample, int nTotal, int nSample); // randomly extract nSample data points from the original data file and put them in sample file

void multi_gaussian_random(vec &mean, mat &cov, int N, mat &res); // generate N samples independently from N(mean, cov)
void multi_gaussian_random(vec &mean, mat &cov, vec &x); // sample a multivariate sample from N(mean, cov)
void grand(double mean, double sigma, int N, vec &x); // generate n samples from N(mean, sigma^2)
double grand();

void vectorise(mat &theta, vec &s, vec &res);
void unvectorise(mat &theta, vec &s, vec &res, int t_r, int t_c);

#endif /* LIBHEAD_H_ */
