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

    def rollout_step(self, obs, critic_obs, amp_obs):
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
        if self.alg.actor_critic.discriminator is not None:
            next_amp_obs = infos["observations"]["amp_policy"]
            next_amp_obs = next_amp_obs.to(self.device)
            amp_obs = amp_obs.to(self.device)
            rewards = self.alg.actor_critic.discriminator.predict_amp_reward(amp_obs, next_amp_obs, rewards)
        # process the step
        self.alg.process_env_step(rewards, dones, infos, amp_obs, next_amp_obs)
        amp_obs = torch.clone(next_amp_obs)
        return obs, critic_obs, rewards, dones, infos, amp_obs

