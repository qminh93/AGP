#include "configuration.h"

configuration::configuration()
{
    // Do nothing
}

configuration::~configuration()
{
    // Free memory (if any)
}

void configuration::load_config(char* config_file)
{
     ifstream cfg(config_file);
     string header, comment;

     while (getline(cfg, header,'\n'))
     {
         cout << header << endl;
         // Extracting comments
         comment = CM;
         while (comment.substr(0, 2) == CM)
         {
             getline(cfg, comment,'\n');
             if (comment.substr(0, 2) == CM) cout << comment << endl;
         }
         stringstream input(comment);
         // Loading contents
         if (header == DS)
         {
            input >> this->data_file >> this->result_file;
            cout << this->data_file << " " << this->result_file << endl;
            cfg >> this->nPoint >> this->nBlock >> this->nBand;
            cfg >> this->support_per_block >> this->max_support >> this->pTest;
            cout << this->nPoint << " " << this->nBlock << " " << this->nBand << endl;
            cout << this->support_per_block << " " << this->max_support << " " << this->pTest << endl;
            getline(cfg, comment,'\n');
         }
         else if (header == BS)
         {
             input >> this->nBasis;
             cout << this->nBasis << endl;
         }
         else if (header == HS)
         {
             input >> this->h_alpha >> this->h_beta >> this->h_gamma;
             cout << this->h_alpha << " " << this->h_beta << " " << this->h_gamma << endl;
         }
         else if (header == VS)
         {
             input >> this->alpha >> this->beta >> this->gamma;
             cout << this->alpha << " " << this->beta << " " << this->gamma << endl;
         }
         else if (header == LS)
         {
             input >> this->training_num_ite; cout << this->training_num_ite << endl;
             cfg >> this->b_r >> this->b_tau >> this->b_lambda >> this->b_decay;
             cout << this->b_r << " " << this->b_tau << " " << this->b_lambda << " " << this->b_decay << endl;
             cfg >> this->M_r >> this->M_tau >> this->M_lambda >> this->M_decay;
             cout << this->M_r << " " << this->M_tau << " " << this->M_lambda << " " << this->M_decay << endl;
             getline(cfg, comment,'\n');
         }
         else if (header == PS)
         {
             input >> this->anytime_interval >> this->anytime_k_sample >> this->anytime_z_sample;
             cout << this->anytime_interval << " " << this->anytime_k_sample << " " << this->anytime_z_sample << endl;
         }
         else
         {
             cout << "BAD CONFIGURATION FILE ! NOTHING CAN BE DONE :(" << endl;
             return;
         }
     }
     cfg.close();
}
