#include "libhead.h"
#include "configuration.h"
#include "optimizer.h"
#include "raw.h"

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
    od->noise = 0.25; od->signal = 7.0;
    od->process(raw, config->nBlock, config->pTest, config->support_per_block, config->max_support);
    cout << "Done." << endl; // TESTED: organizing data module is bug-free
    // commence learning ...
    cout << "Commence anytime learning ..." << endl;
    optimizer* learner = new optimizer(od, config, config->nBasis);
    learner->optimize();
    cout << "Done." << endl;
}

int main()
{
    test();
    return 0;
}
