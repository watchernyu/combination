from quick_plot_helper import quick_plot

twocolordoulbe = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange',]
twosoliddashed = ['dashed', 'dashed',  'solid', 'solid', ]
threecolordoulbe = ['tab:blue', 'tab:orange', 'tab:red', 'tab:blue', 'tab:orange', 'tab:red']
threesoliddashed = ['dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid', ]
standard_6_colors = ('tab:red', 'tab:orange', 'tab:blue', 'tab:brown', 'tab:pink','tab:grey')

envs = ['Hopper-v3', 'Walker2d-v3', 'Ant-v3', 'Humanoid-v3']
data_path = '../data/'

standard_ys = ['AverageTestEpRet', 'AverageQ1Vals', 'AverageNormQBias', 'StdNormQBias', 'Time']

plot_proj1 = True
if plot_proj1:
    quick_plot( # labels, folder names, envs
        [
            'REDQ',
            'ensdecay10K',
            'ensdecay20K',
        ],
        [
            'REDQv2baseline',
            'e_e300_q10_uf20_ed10000',
            'e_e300_q10_uf20_ed20000',
        ],
        envs=envs,
        save_name='REDQ_baseline',
        base_data_folder_path=data_path,
        y_value=standard_ys
    )

