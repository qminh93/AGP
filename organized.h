#ifndef ORGANIZEDDATA_H
#define ORGANIZEDDATA_H

#include "libhead.h"
#include "raw.h"
#include "kmean.h"

class OrganizedData
{
    public:

        int nTrain, nTest, nSupport, nDim, nBlock, bSize, tSize;
        bmat train, test, support, ytrain_t;

        double noise, signal, y_mean;

        OrganizedData();
       ~OrganizedData();

        void    process(RawData* raw, int nBlock, double pTest, int support_per_block, int max_number_support);

        mat     *getxb (int &i);   // i-th training input data block
        mat     *getyb (int &i);   // i-th training output vector
        mat		*getybt(int &i);   // i-th training output vector transpose
        mat     *getxt (int &i);   // i-th testing input data block
        mat     *getyt (int &i);   // i-th testing output vector
        mat     *getxm ();         // inducing input
        mat     *getym ();         // inducing output

    protected:
    private:
};

#endif // ORGANIZEDDATA_H
