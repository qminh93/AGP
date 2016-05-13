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
    SEED(SEED_DEFAULT);
    // load the configuration
    cout << "Loading experiment settings ..." << endl;
    configuration* config = new configuration();
    config->load_config("sample_config.txt");
    cout << "Done." << endl; // TESTED: configuration reading is bug-free
    SEED(SEED_DEFAULT);
    // organize data
    cout << "Loading data ..." << endl;
    OrganizedData* od = new OrganizedData();
    od->noise = 2.0; od->signal = 7.0; od->gamma = 0.01;
    if (!config->preload)
    {
        cout << "Loading raw data ..." << endl;
        RawData* raw = new RawData(config->nPoint);
        raw->load(config->data_file.c_str());
        cout << "Done." << endl;
        od->process(raw, config->nBlock, config->pTest);
        od->save_preset_partition(config->par_file.c_str());
    }
    else od->load_preset_partition(config->nDim, config->nBlock, config->par_file.c_str());
    cout << "Done." << endl; // TESTED: organizing data module is bug-free

    // commence learning ...
    cout << "Commence anytime learning ..." << endl;
    SEED(SEED_DEFAULT);

    vvd res_ave(5, vd(1 + (config->training_num_ite / config->anytime_interval), 0.0));
    vvd res_dev(5, vd(1 + (config->training_num_ite / config->anytime_interval), 0.0));
    vector <vvd> res(config->nSeed);

    SFOR(i, config->nSeed)
    {
        cout << "Training sequence for seed " << config->seed[i] << endl;

        SEED(config->seed[i]);
        optimizer* learner = new optimizer(od, config, config->nBasis);
        learner->optimize();

        string seed_res = "result_" + NUM2STR(config->seed[i]) + ".csv";
        ofstream result(seed_res.c_str());

        cout << "Saving results ..." << endl;
        res[i] = learner->res;
        SFOR(t, learner->res[INDX].size())
        {
            SFOR(pos, PTIM) result << learner->res[pos][t] << ",";
            result << learner->res[PTIM][t] << endl;
            SFOR(pos, PTIM + 1) res_ave[pos][t] += (1.0 / config->nSeed) * learner->res[pos][t];
        }
        result.close();
    }

    cout << "Averaging results ..." << endl;
    ofstream result(config->result_file.c_str());

    SFOR(pos, PTIM + 1)
    SFOR(t, res_ave[INDX].size())
    {
        SFOR(i, config->nSeed) res_dev[pos][t] += (1.0 / config->nSeed) * SQR(res[i][pos][t] - res_ave[pos][t]);
        res_dev[pos][t] = sqrt(res_dev[pos][t]);
    }

    cout << "Saving averaged results ..." << endl;

    SFOR(t, res_ave[INDX].size())
    {
        result << res_ave[INDX][t] << "," << res_ave[TIME][t] << "," << res_ave[RMSE][t] << ","
               << res_ave[DIAG][t] << "," << res_dev[RMSE][t] << "," << res_dev[DIAG][t] << "," << res_ave[PTIM][t] << endl;
    }

    result.close();
    cout << "Done." << endl;
}

int main()
{
    test();
    return 0;
}
