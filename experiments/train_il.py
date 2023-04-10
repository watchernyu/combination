import numpy as np
import torch
import gym
import d4rl
import time
import sys
from redq.algos.cql import CQLAgent
from redq.algos.il import ILAgent
from redq.algos.core import mbpo_epoches, test_agent
from redq.utils.run_utils import setup_logger_kwargs
from redq.utils.bias_utils import log_bias_evaluation
from redq.utils.logx import EpochLogger

def train_d4rl(env_name='hopper-expert-v2', seed=0, epochs=200, steps_per_epoch=1000,
               max_ep_len=1000, n_evals_per_epoch=1,
               logger_kwargs=dict(), debug=False,
               # following are agent related hyperparameters
               hidden_layer=2, hidden_unit=256,
               replay_size=int(1e6), batch_size=256,
               lr=3e-4, gamma=0.99, polyak=0.995,
               alpha=0.2, auto_alpha=True, target_entropy='mbpo',
               start_steps=5000, delay_update_steps='auto',
               utd_ratio=20, num_Q=10, num_min=2, q_target_mode='min',
               policy_update_delay=20,
               # following are bias evaluation related
               evaluate_bias=False, n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True,
               # new experiments
               ensemble_decay_n_data=20000, safe_q_target_factor=0.5,
               do_pretrain=False,
               ):
    """
    :param env_name: name of the gym environment
    :param seed: random seed
    :param epochs: number of epochs to run
    :param steps_per_epoch: number of timestep (datapoints) for each epoch
    :param max_ep_len: max timestep until an episode terminates
    :param n_evals_per_epoch: number of evaluation runs for each epoch
    :param logger_kwargs: arguments for logger
    :param debug: whether to run in debug mode
    :param hidden_layer: number of hidden layers
    :param hidden_unit: hidden layer number of units
    :param replay_size: replay buffer size
    :param batch_size: mini-batch size
    :param lr: learning rate for all networks
    :param gamma: discount factor
    :param polyak: hyperparameter for polyak averaged target networks
    :param alpha: SAC entropy hyperparameter
    :param auto_alpha: whether to use adaptive SAC
    :param target_entropy: used for adaptive SAC
    :param start_steps: the number of random data collected in the beginning of training
    :param delay_update_steps: after how many data collected should we start updates
    :param utd_ratio: the update-to-data ratio
    :param num_Q: number of Q networks in the Q ensemble
    :param num_min: number of sampled Q values to take minimal from
    :param q_target_mode: 'min' for minimal, 'ave' for average, 'rem' for random ensemble mixture
    :param policy_update_delay: how many updates until we update policy network
    """
    hidden_sizes = [hidden_unit for _ in range(hidden_layer)]
    if debug: # use --debug for very quick debugging
        for _ in range(3):
            print("!!!!USING DEBUG SETTINGS!!!!")
        hidden_sizes = [2,2]
        batch_size = 2
        utd_ratio = 2
        num_Q = 3
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100
        epochs = 5

    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set number of epoch
    if epochs == 'mbpo' or epochs < 0:
        epochs = mbpo_epoches[env_name]
    n_offline_updates = steps_per_epoch * epochs + 1

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    #     logger_kwargs = dict(output_dir=osp.join(data_dir, relpath),
    #                          exp_name=exp_name)
    logger_kwargs['output_fname'] = 'pretrain_progress.txt'
    pretrain_logger = EpochLogger(**logger_kwargs)

    """set up environment and seeding"""
    env_fn = lambda: gym.make(env_name)
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()
    # seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seed environment along with env action space so that everything is properly seeded for reproducibility
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)
    seed_all(epoch=0)

    """prepare to init agent"""
    # get obs and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # if environment has a smaller max episode length, then use the environment's max episode length
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()
    # flush logger (optional)
    sys.stdout.flush()
    #################################################################################################

    """load data here"""
    dataset = d4rl.qlearning_dataset(env)
    print("Env: %s, number of data loaded: %d." % (env_name, dataset['actions'].shape[0]))

    """init agent and load data into buffer"""
    agent = ILAgent(env_name, obs_dim, act_dim, act_limit, device,
                     hidden_sizes, replay_size, batch_size,
                     lr, gamma, polyak,
                     alpha, auto_alpha, target_entropy,
                     start_steps, delay_update_steps,
                     utd_ratio, num_Q, num_min, q_target_mode,
                     policy_update_delay, ensemble_decay_n_data, safe_q_target_factor)

    agent.load_data(dataset)


    """========================================== pretrain stage =========================================="""
    n_pretrain_updates = 2000 #TODO fix magic number
    pretrain_stage_start_time = time.time()
    if do_pretrain:
        for t in range(n_pretrain_updates):
            agent.pretrain_update(pretrain_logger)

            # End of epoch wrap-up
            if (t+1) % steps_per_epoch == 0:
                epoch = t // steps_per_epoch
                """logging"""
                # Log info about epoch
                time_used = time.time()-pretrain_stage_start_time
                time_hrs = int(time_used / 3600 * 100)/100
                time_total_est_hrs = (n_pretrain_updates/t) * time_hrs
                pretrain_logger.log_tabular('Epoch', epoch)
                pretrain_logger.log_tabular('TotalEnvInteracts', t)
                pretrain_logger.log_tabular('Time', time_used)
                pretrain_logger.log_tabular('LossPretrain', with_min_and_max=True)
                pretrain_logger.log_tabular('Hours', time_hrs)
                pretrain_logger.log_tabular('TotalHoursEst', time_total_est_hrs)
                pretrain_logger.dump_tabular()

                # flush logged information to disk
                sys.stdout.flush()

        time_used = time.time() - pretrain_stage_start_time
        time_hrs = int(time_used / 3600 * 100) / 100
        print('Pretraining finished in %.2f hours.' % time_hrs)
        print('Saved to %s' % pretrain_logger.output_file.name)

    """========================================== offline stage =========================================="""
    n_offline_updates = 2000 #TODO fix magic number
    # keep track of run time
    offline_stage_start_time = time.time()
    for t in range(n_offline_updates):
        agent.update(logger)

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent(agent, test_env, max_ep_len, logger) # add logging here
            if evaluate_bias:
                log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)

            # reseed should improve reproducibility (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)

            """logging"""
            # Log info about epoch
            time_used = time.time()-offline_stage_start_time
            time_hrs = int(time_used / 3600 * 100)/100
            time_total_est_hrs = (n_offline_updates/t) * time_hrs
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time_used)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('Alpha', with_min_and_max=True)
            # logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.log_tabular('Hours', time_hrs)
            logger.log_tabular('TotalHoursEst', time_total_est_hrs)
            logger.dump_tabular()

            # flush logged information to disk
            sys.stdout.flush()
    time_used = time.time() - offline_stage_start_time
    time_hrs = int(time_used / 3600 * 100) / 100
    print('Finished in %.2f hours.' % time_hrs)
    print('Saved to %s' % logger.output_file.name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-expert-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200) # -1 means use mbpo epochs
    parser.add_argument('--exp_name', type=str, default='cql')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # modify the code here if you want to use a different naming scheme
    exp_name_full = args.exp_name + '_%s' % args.env

    # specify experiment name, seed and data_dir.
    # for example, for seed 0, the progress.txt will be saved under data_dir/exp_name/exp_name_s0
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    train_d4rl(args.env, seed=args.seed, epochs=args.epochs,
               logger_kwargs=logger_kwargs, debug=args.debug)
