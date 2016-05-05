#include "organized.h"

OrganizedData::OrganizedData()
{
    nTrain  = nTest  = 0;
    nDim    = nBlock = bSize    = 0;
}

OrganizedData::~OrganizedData()
{
    train.clear(); test.clear();
}

void OrganizedData::save_preset_partition(const char *partition_file)
{
    ofstream par(partition_file);

    par << bSize << "," << tSize << "," << y_mean << endl;

    NFOR(i, j, nBlock, bSize)
    {
        SFOR(t, nDim) par << train(i, 0)->at(j, t) << ",";
        par << train(i, 1)->at(j, 0) << endl;
    }
    NFOR(i, j, nBlock, tSize)
    {
        SFOR(t, nDim) par << test(i, 0)->at(j, t) << ",";
        par << test(i, 1)->at(j, 0) << endl;
    }

    par.close();
}

void OrganizedData::load_preset_partition(int nDim, int nBlock, const char *partition_file)
{
    this->nDim      = nDim;
    this->nBlock    = nBlock;
    train           = bmat (nBlock, 2);
    test            = bmat (nBlock, 2);

    ifstream par(partition_file);
    string line, token;

    getline(par, line);
    ss parse(line);
    getline(parse, token, ',');
    this->bSize = atoi(token.c_str());
    this->nTrain = bSize * nBlock;
    getline(parse, token, ',');
    this->tSize = atoi(token.c_str());
    this->nTest = tSize * nBlock;
    getline(parse, token, ',');
    this->y_mean = atof(token.c_str());

    cout << this->y_mean << endl;
    //cout << bSize << " " << tSize << endl;

    SFOR(i, nBlock)
    {
        mat *XTrain = new mat(bSize, nDim);
        mat *YTrain = new mat(bSize, 1);

        SFOR(j, bSize)
        {
            getline(par, line);
            //cout << line << endl;
            ss parser(line);
            //cout << parser.str() << endl;
            vd temp;
            while (getline(parser, token, ',')) {
                //cout << token << endl;
                temp.push_back(atof(token.c_str()));
            }

            //cout << nDim << " " << temp.size() << endl;

            SFOR(t, nDim) XTrain->at(j, t) = temp[t];
            YTrain->at(j, 0) = temp[nDim];

            temp.clear();
        }
        train(i, 0)  = XTrain;
        train(i, 1)  = YTrain;
    }

    SFOR(i, nBlock)
    {
        mat *XTest = new mat(tSize, nDim);
        mat *YTest = new mat(tSize, 1);

        SFOR(j, tSize)
        {
            getline(par, line);
            ss parser(line);
            vd temp;
            while (getline(parser, token, ','))
                temp.push_back(atof(token.c_str()));

            SFOR(t, nDim) XTest->at(j, t) = temp[t];
            YTest->at(j, 0) = temp[nDim];
            temp.clear();
        }
        test(i, 0)  = XTest;
        test(i, 1)  = YTest;
    }

    par.close();
}

void OrganizedData::standard_process(RawData *raw, int nBlock, double pTest)
{
    int nData = raw->nData, nDim  = raw->nDim - 1;

    this->nBlock    = nBlock;
    this->nDim      = nDim;

    train           = bmat (nBlock, 2);
    test            = bmat (nBlock, 2);

    vec mark(nData); mark.fill(0);

    this->bSize = nData / nBlock;
    this->tSize = (int) floor(bSize * pTest);
    this->nTest = this->tSize * nBlock;
    this->bSize = this->bSize - this->tSize;
    this->nTrain = nData - this->nTest;

    cout << "Extracting " << this->nTest << " testing points ..." << endl;

    vi sample;
    randsample(nData, nTest, sample);
    mat Xtest(nTest, raw->nDim);
    mat Xtrain(nTrain, raw->nDim);

    NFOR(i, j, nTest, raw->nDim) Xtest(i, j) = raw->X(sample[i], j);
    SFOR(i, nTest) mark(sample[i]) = 1; sample.clear();

    cout << "Extracting " << this->nTrain << " training points ..." << endl;

    int pos = 0;
    SFOR(i, nData)
    if (mark[i] == 0)
    {
        SFOR(j, raw->nDim) Xtrain(pos, j) = raw->X(i, j);
        pos = pos + 1;
    }
    mark.clear();

    cout << "Partitioning raw data into " << nBlock << " cluster using K-Mean ..." << endl;

    RawData *trainraw = new RawData(this->nTrain, Xtrain);
    KMean  *partitioner = new KMean(trainraw);
    Partition *clusters = partitioner->cluster(nBlock);

    y_mean = 0.0;
    SFOR(i, nTrain) y_mean += trainraw->X(i, nDim); y_mean /= nTrain;

    cout << "Packaging testing data points into their respective clusters ..." << endl;

    vec selected(this->nTest); selected.fill(0);
    SFOR(i, this->nBlock)
    {
        mat *xt = new mat(this->tSize, nDim),
            *yt = new mat(this->tSize, 1);
        test(i, 0) = xt; test(i, 1) = yt;
    }

    NFOR(i, j, this->nBlock, this->tSize)
    {
        double bestdist = INFTY, dist = INFTY;
        int tpos = -1;

        SFOR(t, this->nTest)
        if (selected(t) < 1)
        {
            dist = 0.0;
            SFOR(v, nDim) dist += SQR(Xtest(t, v) - clusters->C[i][v]);
            if (dist < bestdist) tpos = t;
            bestdist = min(bestdist, dist);
        }

        SFOR(v, nDim) test(i, 0)->at(j, v) = Xtest(tpos, v);
        test(i, 1)->at(j, 0) = Xtest(tpos, nDim); selected(tpos) = 0;
    }

    cout << "Packaging training data points into their respective clusters ..." << endl;
    SFOR(i, this->nBlock)
    {
        cout << "Processing block " << i + 1 << " ..." << endl;
        mat *xb  = new mat(bSize, nDim),
            *yb  = new mat(bSize, 1);
        NFOR(j, t, this->bSize, nDim)
        {
            xb->at(j, t) = trainraw->X(clusters->member[i][j], t);
            yb->at(j, 0) = trainraw->X(clusters->member[i][j], nDim);
        }
        train(i, 0) = xb; train(i, 1) = yb;
        printf("Done ! nTrain[%d] = %d, nTest[%d] = %d .\n", i, (int) train(i, 0)->n_rows, i, (int) test(i, 0)->n_rows);
    }
}

void OrganizedData::process(RawData* raw, int nBlock, double pTest)
{
    int nData = raw->nData, nDim  = raw->nDim - 1;

    this->nBlock    = nBlock;
    this->nDim      = nDim;

    ytrain_t		= bmat (nBlock, 1);
    train           = bmat (nBlock, 2);
    test            = bmat (nBlock, 2);

    vec mark(nData); mark.fill(0);

    cout << "Partitioning raw data into " << nBlock << " cluster using K-Mean ..." << endl;

    KMean  *partitioner = new KMean(raw);
    Partition *clusters = partitioner->cluster(nBlock);

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
