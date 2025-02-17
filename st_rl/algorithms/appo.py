""" PPO with adversarial network """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import st_rl.modules as modules
from st_rl.utils import unpad_trajectories
from st_rl.storage.rollout_storage import ActionLabelRollout, AmpRolloutStorage
from st_rl.algorithms.ppo import PPO
from st_rl.datasets.motion_loader import AMPLoader
from st_rl.modules.normalizer import Normalize

# assuming learning iteration is at an assumable iteration scale
def GET_PROB_FUNC(option, iteration_scale):
    PROB_options = {
        "linear": (lambda x: max(0, 1 - 1 / iteration_scale * x)),
        "exp": (lambda x: max(0, (1 - 1 / iteration_scale) ** x)),
        "tanh": (lambda x: max(0, 0.5 * (1 - torch.tanh(1 / iteration_scale * (x - iteration_scale))))),
    }
    return PROB_options[option]

class APPOAlgoMixin:
    def __init__(self,
            *args,
            **kwargs,
        ):
        """
        Args:
        - distill_latent_obs_component_mapping: a dict of
            {student_obs_component_name: teacher_obs_component_name}
            only when both policy are the instance of EncoderActorCriticMixin
        """
        super().__init__(*args, **kwargs)

        # The shape of normaliz is the same as the observation space of the policy
        # AMP data loader (adversarial motion prior)
        amp_data_cfg = kwargs.get("amp_data_cfg", {})
        self.amp_data = AMPLoader(**amp_data_cfg)
        
        #self.amp_normalizer = Normalize(shape=*self.storage.obs_shape, device=self.device)
        self.amp_normalizer = Normalize()

    def init_storage(self, *args, **kwargs):
        self.transition = AmpRolloutStorage.Transition()
        self.storage = AmpRolloutStorage(
            *args,
            **kwargs,
        )


    def act(self, obs, critic_obs):
        # get actions
        return_ = super().act(obs, critic_obs) # return is transition.actions
        
        #NOTE: fill action labels with actor actions
        self.transition.action_labels = return_

        return return_



    def process_env_step(self, rewards, dones, infos, amp_obs, next_amp_obs):
        self.transition.amp_policy = torch.cat((amp_obs.clone(), next_amp_obs.clone()),dim=-1)
        # get amp data (policy and expert)
        if hasattr(self.actor_critic, "discriminator"):
            num_envs = amp_obs.shape[0]
            self.transition.amp_expert = self.amp_data.sample_expert_obs(num_envs)

        return_ = super().process_env_step(rewards, dones, infos)

        return return_

    
    def compute_losses(self, minibatch):
        losses, inter_vars, stats = super().compute_losses(minibatch)

        # Discriminator loss.
        if hasattr(self.actor_critic, "discriminator"):
            policy_state = minibatch.amp_policy
            expert_state = minibatch.amp_expert
            with torch.no_grad():
                policy_state = self.amp_normalizer(policy_state)
                expert_state = self.amp_normalizer(expert_state)
            policy_d = self.actor_critic.discriminator(policy_state)
            expert_d = self.actor_critic.discriminator(expert_state)
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.actor_critic.discriminator.compute_grad_pen(expert_state, lambda_=10)
            losses["amp_loss"] = amp_loss + grad_pen_loss  

        return losses, inter_vars, stats


class APPO(APPOAlgoMixin, PPO):
    pass
