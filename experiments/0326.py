import os
# Get the current value of LD_LIBRARY_PATH variable
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
# Append the new path to the existing value of LD_LIBRARY_PATH
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
# Set the modified value of LD_LIBRARY_PATH variable
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['PYTHONPATH'] = '/code'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'

print(os.environ.get('LD_LIBRARY_PATH', ''))
print(os.environ.get('PYTHONPATH', ''))
print(os.environ.get('MUJOCO_GL', ''))
print(os.environ.get('MUJOCO_PY_MUJOCO_PATH', ''))

from experiments.train_redq_sac import redq_sac as function_to_run ## here make sure you import correct function
import time
from redq.utils.run_utils import setup_logger_kwargs
from experiments.grid_utils import get_setting_and_exp_name

if __name__ == '__main__':
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    data_dir = '/combdata'

    exp_prefix = 'e'
    # this wi
    settings = ['env_name','',['Hopper-v3', 'HalfCheetah-v3', 'Walker2d-v3', 'Ant-v3', 'Humanoid-v3'],
                'seed','',[0, 1, 2],
                'epochs','e',[300],
                'num_Q','q',[10],
                'utd_ratio','uf',[20],
                'ensemble_decay_n_data', 'ed', [10000, 20000],
                ]
    if args.debug:
        settings = settings + ['debug','debug',[True],]

    indexes, actual_setting, total, exp_name_full = get_setting_and_exp_name(settings, args.setting, exp_prefix)
    print("##### TOTAL NUMBER OF VARIANTS: %d #####" % total)

    logger_kwargs = setup_logger_kwargs(exp_name_full, actual_setting['seed'], data_dir)
    function_to_run(logger_kwargs=logger_kwargs, **actual_setting)
    print("Total time used: %.3f hours." % ((time.time() - start_time)/3600))

"""
before you submit the jobs:
- quick test your code to make sure things will run without bug
- compute the number of jobs, make sure that is consistent with the array number in the .sh file
- in the .sh file make sure you are running the correct python file 
"""
