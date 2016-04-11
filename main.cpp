#include "libhead.h"
#include "configuration.h"
#include "optimizer.h"
#include "raw.h"

void test()
{
    // load the configuration
    cout << "Loading experiment settings ..." << endl;
    configuration* config = new configuration();
    config->load("aimpeak.csv");
    cout << "Done." << endl;
    // load data
    cout << "Loading raw data ..." << endl;
    RawData* raw = new RawData(config->nPoint);
    raw->load(config->data_file);
    cout << "Done." << endl;
    // organize data
    cout << "Organizing data ..." << endl;
    OrganizedData* od = new OrganizedData();
    od->process(raw, config->nBlock, config->pTest, config->support_per_block, config->max_support);
    cout << "Done." << endl;
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
