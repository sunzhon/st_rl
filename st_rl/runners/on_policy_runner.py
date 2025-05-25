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

import torch
import st_rl
import st_rl.algorithms as algorithms
import st_rl.modules as modules
from st_rl.env import VecEnv
from st_rl.utils import ckpt_manipulator
from st_rl.utils import store_code_state

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # get obs num and critic_obs num
        obs, extras = self.env.get_observations()
        self.policy_cfg["num_obs"] = obs.shape[1]

        if "critic" in extras["observations"]:
            self.policy_cfg["num_critic_obs"] = extras["observations"]["critic"].shape[1]
        else:
            self.policy_cfg["num_critic_obs"] = self.policy_cfg["num_obs"]
        
        if self.alg_cfg["class_name"]=="TPPO":
            if "teacher" in extras["observations"]:
                self.alg_cfg["teacher_policy"]["num_actor_obs"] = extras["observations"]["teacher"].shape[1]
                self.alg_cfg["teacher_policy"]["num_critic_obs"] = extras["observations"]["teacher"].shape[1]
            else:
                self.alg_cfg["teacher_policy"]["num_actor_obs"] =  self.policy_cfg["num_critic_obs"]
                self.alg_cfg["teacher_policy"]["num_critic_obs"] =  self.policy_cfg["num_critic_obs"]
            self.alg_cfg["teacher_policy"]["num_actions"] =  self.env.num_actions

        # get actor_critic policy
        actor_critic = modules.build_actor_critic(
            self.env,
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
        ).to(self.device)

        # get algorithm
        alg_class = getattr(algorithms, self.alg_cfg.pop("class_name"))  # PPO
        self.alg: algorithms.PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"] # same variable with the num_transitions_per_env
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env, 
            actor_obs_shape=(self.env.num_obs,), 
            critic_obs_shape=(self.policy_cfg["num_critic_obs"],), 
            action_shape=(self.env.num_actions,),
            #device=self.device,
            **self.cfg["storage"] if "storage" in self.cfg.keys() else {},
        )


        # init normalization
        self.alg.init_obs_norm(self.policy_cfg["num_obs"], self.policy_cfg["num_critic_obs"])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)
        self.git_status_repos = [os.path.dirname(os.path.dirname(st_rl.__file__))]

        _, _ = self.env.reset()


    def init_writter(self, init_at_random_ep_len):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from st_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from stl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        # episode setting
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))


    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False, trial=None):

        # init writer
        self.init_writter(init_at_random_ep_len)

        # get observations
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

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
                for i in range(self.num_steps_per_env):
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

    def evaluation(self):
        self.metrics_velrmsd = self.env.lin_vel_rmsd.mean()
        self.metrics_CoT = self.env.CoT.mean()

    def rollout_step(self, obs, critic_obs, **kwargs):
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

        # process the step
        self.alg.process_env_step(rewards, dones, infos)
        return obs, critic_obs, rewards, dones, infos

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time = time.time() - locs['tot_start_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)

                if "_max" in key:
                    infotensor = infotensor[~infotensor.isnan()]
                    value = torch.max(infotensor) if len(infotensor) > 0 else torch.tensor(float("nan"))
                elif "_min" in key:
                    infotensor = infotensor[~infotensor.isnan()]
                    value = torch.min(infotensor) if len(infotensor) > 0 else torch.tensor(float("nan"))
                else:
                    value = torch.nanmean(infotensor)

                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        for k, v in locs["losses"].items():
            self.writer.add_scalar("Loss/" + k, v.item(), self.current_learning_iteration)
        for k, v in locs["stats"].items():
            self.writer.add_scalar("Train/" + k, v.item(), self.current_learning_iteration)
        
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, self.current_learning_iteration)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), self.current_learning_iteration)
        self.writer.add_scalar('Perf/total_fps', fps, self.current_learning_iteration)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_allocated', torch.cuda.memory_allocated(self.device) / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_global_free_mem', torch.cuda.mem_get_info(self.device)[0] / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_total', torch.cuda.mem_get_info(self.device)[1] / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Train/mean_reward_each_timestep', statistics.mean(locs['rframebuffer']), self.current_learning_iteration)
        self.writer.add_scalar('Eval/mean_velrmsd', self.metrics_velrmsd, self.current_learning_iteration)
        self.writer.add_scalar('Eval/mean_CoT', self.metrics_CoT, self.current_learning_iteration)

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/ratio_above_mean_reward', statistics.mean([(1. if rew > statistics.mean(locs['rewbuffer']) else 0) for rew in locs['rewbuffer']]), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), self.current_learning_iteration)
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
        )
        for k, v in locs["losses"].items():
            log_string += f"""{k:>{pad}} {v.item():.4f}\n"""

        if len(locs['rewbuffer']) > 0:
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                #f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string += (
                f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                #f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                #f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (f"""{'Eval metrics vel rmsd/step:':>{pad}} {self.metrics_velrmsd:.4f}\n""")
        log_string += (f"""{'Eval metrics CoT/step:':>{pad}} {self.metrics_CoT:.4f}\n""")
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n""")
        print(log_string)



    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = self.alg.state_dict() #the tranining policy network state_dict
        saved_dict.update({
            'iter': self.current_learning_iteration,
            'infos': infos,
            })

        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)


    def load(self, path, load_optimizer=True, map_location=None):
        loaded_dict = torch.load(path,map_location=map_location, weights_only=False)

        if self.cfg.get("ckpt_manipulator", False):
            # suppose to be a string specifying which function to use
            print("\033[1;36m Warning: using a hacky way to load the model. \033[0m")
            # merge self.alg.state_dict()["encoder_0"] state in loaded_dict and save to loaded_dict
            loaded_dict = getattr(ckpt_manipulator, self.cfg["ckpt_manipulator"])(
                loaded_dict,
                self.alg.state_dict(),
            )
            print("\033[1;36m Done: using a hacky way to load the model. \033[0m")

        self.alg.load_state_dict(loaded_dict)

        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        self.current_learning_iteration = loaded_dict['iter']
        if self.cfg.get("ckpt_manipulator", False):
            try:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            except:
                print("\033[1;36m Save manipulated checkpoint failed, ignored... \033[0m")

        return loaded_dict['infos']


    def get_inference_policy(self, device=None):
        self.eval_mode() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.alg.empirical_normalization:
            if device is not None:
                self.alg.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.alg.obs_normalizer(x))  # noqa: E731

        return policy


    def train_mode(self):
        self.alg.actor_critic.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        if self.alg.empirical_normalization:
            self.alg.obs_normalizer.train()
            self.alg.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        if self.alg.empirical_normalization:
            self.alg.obs_normalizer.eval()
            self.alg.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
