import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from redq.algos.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer,\
    get_d4rl_target_entropy, TanhGaussianPolicyPretrain

class ILAgent(object):
    """
    imitation learning baseline
    """
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, delay_update_steps='auto',
                 utd_ratio=20, num_Q=10, num_min=2, q_target_mode='min',
                 policy_update_delay=20,
                 ensemble_decay_n_data=20000, # wait for how many data to reduce number of ensemble (e.g. 20,000 data)
                 safe_q_target_factor=0.5,
                 ):
        # set up networks
        self.policy_net = TanhGaussianPolicyPretrain(obs_dim, act_dim, hidden_sizes, action_limit=act_limit).to(device)
        # set up optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # set up replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        # set up other things
        self.mse_criterion = nn.MSELoss()

        # store other hyperparameters
        self.start_steps = start_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_min = num_min
        self.num_Q = num_Q
        self.init_num_Q = num_Q
        self.utd_ratio = utd_ratio
        self.delay_update_steps = self.start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.q_target_mode = q_target_mode
        self.policy_update_delay = policy_update_delay
        self.device = device

        self.ensemble_decay_n_data = ensemble_decay_n_data
        self.max_reward_per_step = 0.1
        self.safe_q_threshold = 100
        self.safe_q_target_factor = safe_q_target_factor

    def __get_current_num_data(self):
        # used to determine whether we should get action from policy or take random starting actions
        return self.replay_buffer.size

    def get_exploration_action(self, obs, env):
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            if self.__get_current_num_data() > self.start_steps:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net.forward(obs_tensor, deterministic=False,
                                             return_log_prob=False)[0]
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action

    def get_test_action(self, obs):
        # given an observation, output a deterministic action in numpy form
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=True,
                                         return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def store_data(self, o, a, r, o2, d):
        # store one transition to the buffer
        self.replay_buffer.store(o, a, r, o2, d)

    def load_data(self, dataset): # load a d4rl q learning dataset
        assert self.replay_buffer.size == 0
        n_data = dataset['actions'].shape[0]
        n_data = min(n_data, self.replay_buffer.max_size)
        self.replay_buffer.obs1_buf[0:n_data] = dataset['observations']
        self.replay_buffer.obs2_buf[0:n_data] = dataset['next_observations']
        self.replay_buffer.acts_buf[0:n_data] = dataset['actions']
        self.replay_buffer.rews_buf[0:n_data] = dataset['rewards']
        self.replay_buffer.done_buf[0:n_data] = dataset['terminals']
        self.replay_buffer.ptr, self.replay_buffer.size = n_data, n_data

    def sample_data(self, batch_size):
        # sample data from replay buffer
        batch = self.replay_buffer.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def update(self, logger):
        # this function is called after each datapoint collected.
        for i_update in range(1):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            """BC loss"""
            # get policy loss
            a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs_tensor)
            policy_loss = F.mse_loss(mean_a_tilda, acts_tensor)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()

            """update networks"""
            self.policy_optimizer.step()

            # by default only log for the last update out of <num_update> updates
            logger.store(LossPi=policy_loss.cpu().item(), LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                         PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

    def pretrain_update(self, logger):
        # predict next obs with current obs
        for i_update in range(1):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            """mse loss for predicting next obs"""
            obs_next_pred = self.policy_net.predict_next_obs(obs_tensor)
            pretrain_loss = F.mse_loss(obs_next_pred, obs_next_tensor)
            self.policy_optimizer.zero_grad()
            pretrain_loss.backward()

            """update networks"""
            self.policy_optimizer.step()

            # by default only log for the last update out of <num_update> updates
            logger.store(LossPretrain=pretrain_loss.cpu().item())
