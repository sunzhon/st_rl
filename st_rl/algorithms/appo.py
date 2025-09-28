""" PPO with adversarial network """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import st_rl.modules as modules
from st_rl.utils import unpad_trajectories
from st_rl.storage.rollout_storage import ActionLabelRollout, AmpRolloutStorage
from st_rl.algorithms.ppo import PPO
from st_rl.algorithms.tppo import TPPO
from st_rl.datasets.motion_loader import AMPLoader
from st_rl.modules.normalizer import Normalizer
from st_rl.modules import EmpiricalNormalization


# assuming learning iteration is at an assumable iteration scale
def GET_PROB_FUNC(option, iteration_scale):
    PROB_options = {
        "linear": (lambda x: max(0, 1 - 1 / iteration_scale * x)),
        "exp": (lambda x: max(0, (1 - 1 / iteration_scale) ** x)),
        "tanh": (
            lambda x: max(
                0, 0.5 * (1 - torch.tanh(1 / iteration_scale * (x - iteration_scale)))
            )
        ),
    }
    return PROB_options[option]


class APPOAlgoMixin:
    def __init__(
        self,
        *args,
        discriminator_loss_coef=1.0,
        bc_loss_coef=0.0,
        amp_obs_dim=None,
        **kwargs,
    ):
        """
        Args:
        - distill_latent_obs_component_mapping: a dict of
            {student_obs_component_name: teacher_obs_component_name}
            only when both policy are the instance of EncoderActorCriticMixin
        """
        super().__init__(*args, **kwargs)

        # The shape of the normalizer is the same as the observation space
        
        self.discriminator_loss_coef = discriminator_loss_coef
        self.bc_loss_coef = bc_loss_coef
        # Initialize the AMP normalizer with the observation shape and device
        #self.ref_normalizer = Normalizer(kwargs["amp_obs_dim"])
        self.ref_obs_normalizer = EmpiricalNormalization(shape=[amp_obs_dim], until=1.0e8).to(self.device)
        self.ref_obs_normalizer.train()
        self.ref_motion_normalizer = EmpiricalNormalization(shape=[amp_obs_dim], until=1.0e8).to(self.device)
        self.ref_motion_normalizer.train()

    def init_storage(self, *args, **kwargs):
        self.transition = AmpRolloutStorage.Transition()
        self.storage = AmpRolloutStorage(
            *args,
            **kwargs,
        )

    def act(self, obs, critic_obs):
        # get actions
        return_ = super().act(obs, critic_obs)  # return is transition.actions

        # fill action labels with actor actions
        self.transition.action_labels = return_

        return return_

    def process_env_step(self, rewards, dones, infos):

        # get amp ref obs
        assert "observations" in infos and "ref_obs" in infos["observations"], "Missing ref_obs in infos"
        ref_obs = infos["observations"]["ref_obs"]
        B = ref_obs.shape[0]
        ref_obs = ref_obs.reshape(B, -1).to(self.device)
        ref_obs = self.ref_obs_normalizer(ref_obs)

        ref_motion = infos["ref_motion"]
        ref_motion = self.ref_motion_normalizer(ref_motion)

        # add ref obs and ref motion in transition
        self.transition.ref_obs = ref_obs
        self.transition.ref_motion = ref_motion

        # update style rewards and task rewards
        style_reward, task_reward, rewards = self.actor_critic.discriminator.predict_style_reward(ref_obs, rewards)
        infos["log"]["Episode_Reward/style_reward"] = style_reward.mean().item()
        infos["log"]["Episode_Reward/task_reward"] = task_reward.mean().item()
        ref_reward, _, _ = self.actor_critic.discriminator.predict_style_reward(ref_motion, rewards)
        infos["log"]["Episode_Reward/ref_reward"] = ref_reward.mean().item()
        infos["log"]["Episode_Reward/reward"] = rewards.mean().item()

        return_ = super().process_env_step(rewards, dones, infos)

        return return_

    def compute_losses(self, minibatch):
        losses, inter_vars, stats = super().compute_losses(minibatch)

        if hasattr(self, "dagger_loss_coef"):
            if self.dagger_loss_coef != 0:
                # this has problem
                dagger_loss = torch.norm(
                    self.actor_critic.action_mean - minibatch.action_labels, dim=-1
                ).mean()
                losses["dagger_loss"] = dagger_loss

        if hasattr(self, "bc_loss_coef"):
            if self.bc_loss_coef != 0:
                policy_state = minibatch.ref_obs
                expert_state = minibatch.ref_motion
                bc_loss = torch.norm(
                    policy_state - expert_state, dim=-1
                ).mean()
                losses["bc_loss"] = bc_loss

        # Discriminator loss.
        if hasattr(self.actor_critic, "discriminator"):
            if self.discriminator_loss_coef != 0:
                policy_state = minibatch.ref_obs
                expert_state = minibatch.ref_motion
                policy_d = self.actor_critic.discriminator(policy_state)
                expert_d = self.actor_critic.discriminator(expert_state)
                expert_loss = torch.nn.MSELoss()(
                    expert_d, torch.ones(expert_d.size(), device=self.device)
                )
                policy_loss = torch.nn.MSELoss()(
                    policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
                )
                grad_pen_loss = self.actor_critic.discriminator.compute_grad_pen(expert_state)
                losses["discriminator_expert_loss"] = 0.5*expert_loss
                losses["discriminator_policy_loss"] = 0.5*policy_loss
                losses["discriminator_grad_len_loss"] = grad_pen_loss

        return losses, inter_vars, stats


class APPO(APPOAlgoMixin, PPO):
    """
    APPO (Adversarial Proximal Policy Optimization) class that combines 
    the functionalities of APPOAlgoMixin and PPO. This class is used for 
    training policies with adversarial motion priors.
    
    Inherits:
        APPOAlgoMixin: Mixin class providing adversarial training methods.
        PPO: Proximal Policy Optimization algorithm class.
    """
    pass


class ATPPO(APPOAlgoMixin, TPPO):
    """
    APPO (Adversarial Proximal Policy Optimization) class that combines 
    the functionalities of APPOAlgoMixin and PPO. This class is used for 
    training policies with adversarial motion priors.
    
    Inherits:
        APPOAlgoMixin: Mixin class providing adversarial training methods.
        PPO: Proximal Policy Optimization algorithm class.
    """
    pass
