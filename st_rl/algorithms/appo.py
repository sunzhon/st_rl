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
from st_rl.modules.normalizer import Normalizer


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
        # of the policy
        # AMP data loader (adversarial motion prior)
        amp_data_cfg = kwargs.get("amp_data", {})
        amp_data_cfg["device"] = self.device
        self.amp_data = AMPLoader(**amp_data_cfg)

        if "mimic_loss_coef" in amp_data_cfg.keys():
            self.mimic_loss_coef = amp_data_cfg["mimic_loss_coef"]
            print(f"mimic loss: {self.mimic_loss_coef}")
        if "discriminator_loss_coef" in amp_data_cfg.keys():
            self.discriminator_loss_coef = amp_data_cfg["discriminator_loss_coef"]

        # Initialize the AMP normalizer with the observation shape and device
        self.amp_normalizer = Normalizer(amp_data_cfg["amp_obs_dim"])

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

    def process_env_step(self, rewards, dones, infos, amp_obs):
        # get amp policy data
        self.transition.amp_policy = amp_obs.clone()
        # get amp expert data
        self.transition.amp_expert = self.expert_obs_generator.__next__()
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

        if hasattr(self, "mimic_loss_coef"):
            if self.mimic_loss_coef != 0:
                policy_state = minibatch.amp_policy
                expert_state = minibatch.amp_expert
                mimic_loss = torch.norm(
                    policy_state - expert_state, dim=-1
                ).mean()
                losses["mimic_loss"] = mimic_loss

        # Discriminator loss.
        if hasattr(self.actor_critic, "discriminator"):
            if self.discriminator_loss_coef != 0:
                policy_state = minibatch.amp_policy
                expert_state = minibatch.amp_expert
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(
                        policy_state, self.device
                    )
                    expert_state = self.amp_normalizer.normalize_torch(
                        expert_state, self.device
                    )
                policy_d = self.actor_critic.discriminator(policy_state)
                expert_d = self.actor_critic.discriminator(expert_state)
                expert_loss = torch.nn.MSELoss()(
                    expert_d, torch.ones(expert_d.size(), device=self.device)
                )
                policy_loss = torch.nn.MSELoss()(
                    policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
                )
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.actor_critic.discriminator.compute_grad_pen(expert_state) # expert_state was normalized first
                losses["discriminator_loss"] = amp_loss + grad_pen_loss

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state.detach().cpu().numpy())
                    self.amp_normalizer.update(expert_state.detach().cpu().numpy())

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
