import os.path
import numpy as np
import pandas as pd
import json
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

def get_extra_dict_multiple_seeds(datafolder_path): # obtain measures for a alg-dataset variant
    if not os.path.exists(datafolder_path):
        raise FileNotFoundError("Path does not exist: %s" % datafolder_path)
    # return a list, each entry is the final performance of a seed
    final_test_return_seeds = []
    final_test_normalized_return_seeds = []
    weight_diff_seeds = []
    feature_diff_seeds = []
    for subdir, dirs, files in os.walk(datafolder_path):
        if 'extra.json' in files:
            extra_dict_file_path = os.path.join(subdir, 'extra.json')
            with open(extra_dict_file_path, 'r') as file:
                extra_dict = json.load(file)
                final_test_return_seeds.append(np.mean(extra_dict['final_test_returns']))
                final_test_normalized_return_seeds.append(np.mean(extra_dict['final_test_normalized_returns']))
                weight_diff_seeds.append(extra_dict['weight_diff'])
                feature_diff_seeds.append(extra_dict['feature_diff'])

    extra_dict = {}
    extra_dict['performance'] = [np.mean(final_test_return_seeds), np.std(final_test_return_seeds)]
    extra_dict['performance_std'] = [np.std(final_test_return_seeds),]
    extra_dict['performance_normalized']= [np.mean(final_test_normalized_return_seeds), np.std(final_test_normalized_return_seeds)]
    extra_dict['performance_normalized_std'] = [np.std(final_test_normalized_return_seeds),]
    extra_dict['weight_diff']= [np.mean(weight_diff_seeds), np.std(weight_diff_seeds)]
    extra_dict['feature_diff']= [np.mean(feature_diff_seeds), np.std(feature_diff_seeds)]
    return extra_dict

data_path = '../data/il/'
datasets = [
    'halfcheetah-medium-v2', 'halfcheetah-medium-expert-v2', 'halfcheetah-medium-replay-v2',
    'hopper-medium-v2', 'hopper-medium-expert-v2', 'hopper-medium-replay-v2',
    'walker2d-medium-v2', 'walker2d-medium-expert-v2', 'walker2d-medium-replay-v2',
]

# final table: for each variant name, for each measure, compute relevant values
alg_dataset_dict = {}
algs = ['il_preFalse', 'il_preTrue']
for alg in algs:
    alg_dataset_dict[alg] = {}
    # compute a number of measures
    # 1. performance
    performance_list = []
    performance_std_list = []
    for dataset in datasets:
        folderpath = os.path.join(data_path, '%s_%s' % (alg, dataset))
        alg_dataset_dict[alg][dataset] = get_extra_dict_multiple_seeds(folderpath)

# TODO compute performance gain from pretraining for ones use pretraining (compared to no pretrain baseline)

def get_aggregated_value(alg_dataset_dict, alg, measure):
    value_list = []
    for dataset, extra_dict in alg_dataset_dict[alg].items():
        value_list.append(extra_dict[measure][0]) # each entry is the value from a dataset
    return np.mean(value_list), np.std(value_list)

val = get_aggregated_value(alg_dataset_dict, 'il_preFalse', 'performance_normalized')

"""table generation"""
rows = ['performance_normalized', 'performance_normalized_std', 'weight_diff', 'feature_diff']
row_names = ['Performance', 'Performance std over seeds', 'Change in weight', 'Change in feature']

# each time we generate a row
for measure, row_name in zip(rows, row_names):
    row_string = row_name
    for alg in algs:
        val = get_aggregated_value(alg_dataset_dict, alg, measure)
        row_string += (' & %.3f $\pm$ %.3f' % val)
    row_string += '\\\\'
    print(row_string)



# first compute everything, put them into a dictionary... nested dictionary? first key alg then measure
# when drawing table, specify the rows and column headers, then for each entry, simply look it up in the dictionary.




