import os.path
import numpy as np
import pandas as pd
from quick_plot_helper import quick_plot

def get_final_performance_seeds(datafolder_path):
    if not os.path.exists(datafolder_path):
        raise FileNotFoundError("Path does not exist: %s" % datafolder_path)
    # return a list, each entry is the final performance of a seed
    performance_list = []
    for subdir, dirs, files in os.walk(datafolder_path):
        if 'progress.txt' in files:
            # load progress file for this seed
            progress_file_path = os.path.join(subdir, 'progress.txt')
            df = pd.read_table(progress_file_path)
            final_performance = df['AverageTestEpRet'].tail(2).mean()
            performance_list.append(final_performance)
    return performance_list

data_path = '../data/il/'
datasets = ['halfcheetah-medium-expert-v2', 'halfcheetah-medium-replay-v2']

# final table: for each variant name, for each measure, compute relevant values
algs = ['il_preFalse_debugTrue', 'il_preTrue_debugTrue']
for alg in algs:
    # compute a number of measures
    # 1. performance
    performance_list = []
    performance_std_list = []
    for dataset in datasets:
        folderpath = os.path.join(data_path, '%s_%s' % (alg, dataset))
        performance_seeds = get_final_performance_seeds(folderpath)
        performance_average_over_seeds = np.mean(performance_seeds)
        performance_std_over_seeds = np.std(performance_seeds)
        performance_list.append(performance_average_over_seeds)
        performance_std_list.append(performance_std_over_seeds)
    alg_average_performance = np.mean(performance_list)
    alg_average_std = np.mean(performance_std_list)

