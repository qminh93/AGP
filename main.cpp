#include "libhead.h"
#include "configuration.h"
#include "optimizer.h"
#include "raw.h"

#define RMSE 2
#define DIAG 3
#define TIME 1
#define PTIM 4
#define INDX 0

void test()
{
    // load the configuration
    cout << "Loading experiment settings ..." << endl;
    configuration* config = new configuration();
    config->load_config("sample_config.txt");
    cout << "Done." << endl; // TESTED: configuration reading is bug-free
    // load data
    cout << "Loading raw data ..." << endl;
    RawData* raw = new RawData(config->nPoint);
    raw->load(config->data_file.c_str());
    //raw->normalize_x();
    cout << "Done." << endl; // TESTED: loading raw data module is bug-free
    // organize data
    cout << "Organizing data ..." << endl;
    OrganizedData* od = new OrganizedData();
    od->noise = 0.25; od->signal = 4.0;
    if (!config->preload)
    {
        od->process(raw, config->nBlock, config->pTest);
        od->save_preset_partition(config->par_file.c_str());
    }
    else
    {
        cout << config->par_file << endl;
        od->load_preset_partition(config->nDim, config->nBlock, config->par_file.c_str());
    }
    cout << "Done." << endl; // TESTED: organizing data module is bug-free
    // commence learning ...
    cout << "Commence anytime learning ..." << endl;
    cout << config->nSeed << endl;
    vvd res_ave(5);
    vvd res_dev(5);
    vector <vvd> res(config->nSeed);

    SFOR(i, config->nSeed)
    {
        cout << "Training sequence for seed " << config->seed[i] << endl;
        SEED(config->seed[i]);
        optimizer* learner = new optimizer(od, config, config->nBasis);
        learner->optimize();

        string hyper    = "hyp_learned_" + NUM2STR(config->seed[i]) + ".bin";
        string seed_res = "result_" + NUM2STR(config->seed[i]) + ".csv";
        string ave_res  = "result_ave.csv";

        learner->savedItems.save(hyper.c_str());
        res[i] = learner->res;

        if (i == 0) SFOR(j, 5)
        {
            res_ave[j] = vd(learner->res[j].size(), 0.0);
            res_dev[j] = vd(learner->res[j].size(), 0.0);
        }

        ofstream result(seed_res.c_str());
        SFOR(t, learner->res[0].size())
        {
            SFOR(j, 4)
            {
                result << learner->res[j][t];
                if (j < 3) result << ",";
                else result << endl;

                res_ave[j][t] += (1.0 / config->nSeed) * learner->res[j][t];
            }
        }
        result << learner->res[PTIM][0] << endl;
        res_ave[PTIM][0] += (1.0 / config->nSeed) * learner->res[PTIM][0];
        result.close();

    }

    ofstream result(config->result_file.c_str());

    SFOR(t, res_ave[0].size())
    {
        SFOR(j, 4)
        {
            SFOR(i, config->nSeed)
                res_dev[j][t] += (1.0 / config->nSeed) * SQR(res[i][j][t] - res_ave[j][t]);
            res_dev[j][t] = sqrt(res_dev[j][t]);

            result << res_ave[j][t];
            if (j < 3) result << ",";
            else result << endl;
        }
        SFOR(j, 4)
        {
            result << res_dev[j][t];
            if (j < 3) result << ",";
            else result << endl;
        }
    }
    SFOR(j, 4)
    {
        result << res_ave[PTIM][0];
        if (j < 3) result << ",";
        else result << endl;
    }
    result.close();
    cout << "Done." << endl;
}

int main()
{
    test();
    //od_test();
    return 0;
}
