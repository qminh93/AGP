#include "organized.h"

OrganizedData::OrganizedData()
{
    nTrain  = nTest  = nSupport = 0;
    nDim    = nBlock = bSize    = 0;
}

OrganizedData::~OrganizedData()
{
    train.clear(); test.clear(); support.clear();
}

void OrganizedData::process(RawData* raw, int nBlock, double pTest, int support_per_block, int max_number_support)
{
    int nData = raw->nData, nDim  = raw->nDim - 1;

    this->nSupport  = support_per_block * nBlock;
    this->nBlock    = nBlock;
    this->nDim      = nDim;

    ytrain_t		= bmat (nBlock, 1);
    train           = bmat (nBlock, 2);
    test            = bmat (nBlock, 2);
    support         = bmat (1, 2);

    mat *xm         = new mat(nSupport, nDim),
        *ym         = new mat(nSupport, 1);

    vec mark(nData); mark.fill(0);

    cout << "Partitioning raw data into " << nBlock << " cluster using K-Mean ..." << endl;

    KMean  *partitioner = new KMean(raw);
    Partition *clusters = partitioner->cluster(nBlock);

    cout << "Generating supports from partitioned data" << endl;

    NFOR(i, j, nBlock, support_per_block)
    {
        double mix = 1.0;
        rowvec rv  = raw->X.row(clusters->member[i][j]);
        SFOR(t, nDim) xm->at(i * support_per_block + j, t) = rv(t) * mix + clusters->C[i][t] * (1 - mix);
        ym->at(i * support_per_block + j, 0) = rv(nDim) * mix + clusters->C[i][nDim] * (1 - mix);
    }

    if (max_number_support < this->nSupport) // shrinking the support set
    {
    	vector <int> indices(this->nSupport, 0);
    	SFOR(t, this->nSupport) indices[t] = t;

    	mat *xm_shrink = new mat(max_number_support, nDim);
    	mat *ym_shrink = new mat(max_number_support, 1);

    	int last = this->nSupport - 1, pos;
    	SFOR(t, max_number_support)
    	{
    		pos = IRAND(0, last);
    		SFOR(u, nDim) xm_shrink->at(t, u) = xm->at(indices[pos], u);
    		ym_shrink->at(t, 0) = ym->at(indices[pos], 0);
    		indices[pos] = indices[last]; last--;
    	}

    	delete xm; delete ym;
    	xm = xm_shrink; ym = ym_shrink; this->nSupport = max_number_support;
    }

    support(0, 0) = xm;
    support(0, 1) = ym;

    printf("Done ! nSupport = %d\n", support(0, 0)->n_rows);
    cout << "Packaging training/testing data points into their respective cluster" << endl;

    y_mean = 0.0;
    SFOR(pos, nData) y_mean += raw->X(pos, nDim); y_mean /= nData;

    SFOR(i, nBlock)
    {
        cout << "Processing block " << i + 1 << endl;
        int tSize, pos, counter;

        bSize   = (int) clusters->member[i].size(),
        tSize   = (int) floor(bSize * pTest),
        pos     = 0,
        counter = 0;

        this->tSize = tSize;
        mark    = vec(bSize); mark.fill(0);

        if (bSize > tSize)  // if we can afford to draw tSize test points from this block without depleting it ...
        {
            mat *xt = new mat(tSize, nDim),
                *yt = new mat(tSize, 1);

            SFOR(j, tSize)
            {
                pos = IRAND(0, bSize - 1);
				while (mark[pos])
					pos = IRAND(0, bSize - 1);
				mark[pos] = 1; pos = clusters->member[i][pos];

				SFOR(t, nDim)
				xt->at(j, t) = raw->X(pos, t);
				yt->at(j, 0) = raw->X(pos, nDim);
            }

            bSize     -= tSize; nTest     += tSize;
            test(i, 0) = xt; test(i, 1) = yt;
        }

        nTrain += bSize;

        mat *xb  = new mat(bSize, nDim),
            *yb  = new mat(bSize, 1),
        	*ybt = new mat(1, bSize);

        SFOR(j, (int) mark.n_elem) if (!mark[j])
        {
            SFOR(t, nDim)
            xb ->at(counter,   t) 	= raw->X(clusters->member[i][j], t);
            yb ->at(counter, 0) 	= raw->X(clusters->member[i][j], nDim);
            ybt->at(0, counter)		= yb ->at(counter, 0);
            counter++;
        }

        train(i, 0) 	= xb;
        train(i, 1) 	= yb;
        ytrain_t(i, 0)  = ybt;
        mark.clear();

        printf("Done ! nData[%d] = %d, nTrain[%d] = %d, nTest[%d] = %d .\n",
        i, (int) clusters->member[i].size(), i, train(i, 0)->n_rows, i, (int) test(i, 0)->n_rows);
    }

    bSize = nTrain / nBlock;
}

mat* OrganizedData::getxb(int &i)
{
    return train(i, 0);
}

mat* OrganizedData::getyb(int &i)
{
    return train(i, 1);
}

mat* OrganizedData::getybt(int &i)
{
    return ytrain_t(i, 0);
}

mat* OrganizedData::getxt(int &i)
{
    return test(i, 0);
}

mat* OrganizedData::getyt(int &i)
{
    return test(i, 1);
}

mat* OrganizedData::getxm()
{
    return support(0, 0);
}

mat* OrganizedData::getym()
{
    return support(0, 1);
}
