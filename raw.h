#ifndef RAWDATA_H
#define RAWDATA_H

#define LOAD_LIMIT_DEFAULT 2000000

#include "libhead.h"

class RawData
{
    public:
        mat X; // one example per row
        int nDim, nData, nLim; // nDim = data dimension + 1 -- assuming that the output is always scalar

        RawData();
        RawData(mat& X);
        RawData(int nLim);
        RawData(int nLim, mat &X);
       ~RawData();

        void save(const char* datafile);
        void load(const char* datafile);

        void normalize_x();
        void normalize_y(pdd &meandev);

    protected:
    private:
};

#endif // RAWDATA_H
