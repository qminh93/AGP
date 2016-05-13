#ifndef ORGANIZEDDATA_H
#define ORGANIZEDDATA_H

#include "libhead.h"
#include "raw.h"
#include "kmean.h"

class OrganizedData
{
    public:

        int nTrain, nTest, nDim, nBlock, bSize, tSize;
        bmat train, test, ytrain_t;

        double noise, signal, y_mean, gamma;

        OrganizedData();
       ~OrganizedData();

        void save_preset_partition(const char *partition_file);
        void load_preset_partition(int nDim, int nBlock, const char *partition_file);

        void    process(RawData* raw, int nBlock, double pTest);
        void    standard_process(RawData *raw, int nBlock, double pTest);

        mat     *getxb (int &i);   // i-th training input data block
        mat     *getyb (int &i);   // i-th training output vector
        mat		*getybt(int &i);   // i-th training output vector transpose
        mat     *getxt (int &i);   // i-th testing input data block
        mat     *getyt (int &i);   // i-th testing output vector

    protected:
    private:
};

#endif // ORGANIZEDDATA_H
