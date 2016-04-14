#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "libhead.h"

#define TRAIN 0
#define TEST 1
#define SUPPORT 2
#define HYPER 3

#define DS "/* DATA SETTINGS */"
#define BS "/* BASIS SETTINGS */"
#define HS "/* HYPER SETTINGS */"
#define VS "/* VARIATIONAL SETTINGS */"
#define LS "/* LEARNING SETTINGS */"
#define PS "/* PREDICTION SETTINGS */"
#define CM "//"

class configuration
{
    public:

        /* FILE DIRECTORIES */
        string  data_file, result_file;

        /* DATA SETTINGS */
        int		nBlock, nBand, // the number of partitions and Markov order
                support_per_block, // the number of supporting points to be generated per block
                nPoint, // the maximum number of data points to be loaded in memory
                max_support; // number of support cap
        double  pTest; // percentage test point per block

        /* LEARNING SETTINGS */
        int training_num_ite; // number of training iterations
        double  M_r, M_tau, M_lambda, M_decay,  // learning rate settings for M
                b_r, b_tau, b_lambda, b_decay;  // learning rate settings for b

        /* PREDICTION SETTINGS */
        int		anytime_z_sample, // number of samples z
                anytime_k_sample, // number of sampled Markov clusters
                anytime_interval; // prediction interval

        /* BASIS SETTINGS */
        int nBasis; // number of basis functions

        /* HYPER SETTINGS */
        double  h_alpha, h_beta, h_gamma; // \Theta and \Lambda initialisation parameters

        /* VARIATIONAL SETTINGS */
        double	alpha, beta, gamma;		// M and b initialisation parameters

        configuration();
       ~configuration();

        void load_config(char* config_file);

    protected:
    private:

};

#endif // CONFIGURATION_H
