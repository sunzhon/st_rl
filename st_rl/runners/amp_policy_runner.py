# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from tensorboardX import SummaryWriter
import torch
from st_rl.utils.metrics import *

import st_rl
import st_rl.algorithms as algorithms
import st_rl.modules as modules
from st_rl.env import VecEnv
from st_rl.utils import ckpt_manipulator
from st_rl.utils import store_code_state
from st_rl.runners.on_policy_runner import OnPolicyRunner

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter


class AmpPolicyRunner(OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.alg_cfg.get("amp_data") is not None:
            # for env initialization
            if self.alg_cfg["amp_data"]["init_state_fields"] is not None:
                self.env.unwrapped.scene["robot"].data.demo_init_states = self.alg.amp_data.init_states

            # for amp goal/commands
            if self.alg_cfg["amp_data"]["style_goal_fields"] is not None:
                self.env.unwrapped.scene["robot"].data.style_goal = self.alg.amp_data.style_goal

            # for expressive goal/commands
            if self.alg_cfg["amp_data"]["expressive_goal_fields"] is not None:
                self.env.unwrapped.scene["robot"].data.expressive_goal = self.alg.amp_data.expressive_goal
            
            # for tracking expressive joint fields
            if self.alg_cfg["amp_data"]["expressive_joint_name"] is not None:
                self.env.unwrapped.scene["robot"].data.expressive_tracking_joint_pos = self.alg.amp_data.expressive_joint_pos

            # for tracking expressive link fields
            if self.alg_cfg["amp_data"]["expressive_link_name"] is not None:
                self.env.unwrapped.scene["robot"].data.expressive_tracking_link_pos = self.alg.amp_data.expressive_link_pos
    
    def rollout_step(self, obs, critic_obs):
        # get actions from policy
        actions = self.alg.act(obs, critic_obs)

        # step env
        obs, rewards, dones, infos = self.env.step(actions)
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
        
        # NOTE : terminal_amp_states, see amp for hardware
        # perform normalization
        obs = self.alg.obs_normalizer(obs)
        if "critic" in infos["observations"]:
            critic_obs = self.alg.critic_obs_normalizer(infos["observations"]["critic"])
        else:
            critic_obs = obs

        # process discriminator rewards
        if self.policy_cfg.get("amp_discriminator") is not None:
            amp_obs = infos["observations"]["amp_policy"] # env_num, history_len, field_num
            amp_obs = amp_obs.reshape(amp_obs.shape[0],-1).to(self.device)
            style_reward, task_reward, rewards = self.alg.actor_critic.discriminator.predict_style_reward(amp_obs, rewards, self.alg.amp_normalizer, self.device)
            infos["log"]["Episode_Reward/style_reward"] = style_reward.mean().item()
            infos["log"]["Episode_Reward/task_reward"] = task_reward.mean().item()
            infos["log"]["Episode_Reward/merge_rewards"] = rewards.mean().item()
        else:
            print("Please define amp_discriminator")

        # process the step
        self.alg.process_env_step(rewards, dones, infos, amp_obs)

        # update style/expressive goals and tracking values from the above __nex__() in process_env_step()
        if self.alg_cfg.get("amp_data") is not None:
            # for amp goal/commands
            if self.alg_cfg["amp_data"]["style_goal_fields"] is not None:
                self.env.unwrapped.scene["robot"].data.style_goal = self.alg.amp_data.style_goal
            # for expressive goal/commands
            if self.alg_cfg["amp_data"]["expressive_goal_fields"] is not None:
                self.env.unwrapped.scene["robot"].data.expressive_goal = self.alg.amp_data.expressive_goal
            # for tracking expressive pos fields
            if self.alg_cfg["amp_data"]["expressive_joint_name"] is not None:
                self.env.unwrapped.scene["robot"].data.expressive_tracking_joint_pos = self.alg.amp_data.expressive_joint_pos
            # for tracking expressive link fields
            if self.alg_cfg["amp_data"]["expressive_link_name"] is not None:
                self.env.unwrapped.scene["robot"].data.expressive_tracking_link_pos = self.alg.amp_data.expressive_link_pos


        return obs, critic_obs, rewards, dones, infos



    def learn(self, num_learning_iterations, init_at_random_ep_len=False, trial=None):

        # init writer
        self.init_writter(init_at_random_ep_len)

        # get observations
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        #if "amp_policy" in extras["observations"]:
        #    amp_obs = extras["observations"].get("amp_policy",obs)
        #else:
        #    amp_obs=None

        # switch to training mode
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        print("Initialization done, start learning.")
        print("NOTE: you may see a bunch of `NaN or Inf found in input tensor` once and appears in the log. Just ignore it if it does not affect the performance.")
        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        tot_start_time = time.time()
        start = time.time()
        for it in range(start_iter, tot_iter):
            #1) Rollout, inference_mode for data collection
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                # make an expert obs generator, 
                self.alg.expert_obs_generator = self.alg.amp_data.expert_obs_generator()
                # data collection
                for i in range(self.num_steps_per_env):
                    import pdb;pdb.set_trace()
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rframebuffer.extend(rewards[dones < 1].cpu().numpy().tolist())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            #2) Training, compute loss and update algorithm after an episode
            losses, stats = self.alg.update(self.current_learning_iteration)

            #3) Eval
            self.evaluation()
            if trial is not None:
                trial.report(self.metrics_velrmsd, self.current_learning_iteration)
                trial.report(self.metrics_CoT, self.current_learning_iteration)
        
            #4) Logging
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            if self.current_learning_iteration % self.save_interval == 0 and self.current_learning_iteration > start_iter:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()
            self.current_learning_iteration = self.current_learning_iteration + 1
            start = time.time()

            # store code diff
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        #5) saving training models
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

        return self.metrics_velrmsd, self.metrics_CoT

