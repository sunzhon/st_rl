import numpy as np

import torch
import torch.nn as nn
from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent, ActorCriticHiddenState
from st_rl.modules.mlp import MlpModel
from st_rl.modules.actor_critic_recurrent import Memory
from st_rl.utils import unpad_trajectories
from st_rl.utils.utils import get_subobs_size, get_subobs_by_components, substitute_estimated_state
from st_rl.utils.collections import namedarraytuple

# Define a named tuple for hidden states
EstimatorActorHiddenState = namedarraytuple('EstimatorActorHiddenState', [
    'estimator',
    'actor',
])

class EstimatorMixin:
    def __init__(self,
            *args,
            estimator_obs_components= None, # a list of strings used to get obs slices
            estimator_target_components= None, # a list of strings used to get obs slices
            estimator_kwargs= dict(),
            use_actor_rnn= False, # if set and self.is_recurrent, use actor-rnn output as the input of the state estimator directly
            replace_state_prob= 0., # if 0~1, replace the actor observation with the estimated state with this probability
            **kwargs,
        ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.estimator_obs_components = estimator_obs_components
        self.estimator_target_components = estimator_target_components
        self.estimator_kwargs = estimator_kwargs
        self.use_actor_rnn = use_actor_rnn
        self.replace_state_prob = replace_state_prob
        assert (self.replace_state_prob <= 0.) or (not self.use_actor_rnn), "You cannot replace the actor's observation (part) after the actor already used its memory module."
        self.build_estimator(**kwargs)
        print(f"Estimator: {self.state_estimator}")

    def build_estimator(self, **kwargs):
        """Build the state estimator model."""
        estimator_input_size = get_subobs_size(self.obs_segments, self.estimator_obs_components)
        estimator_output_size = get_subobs_size(self.obs_segments, self.estimator_target_components)
        if self.is_recurrent:
            # Estimate required state using a recurrent network
            if self.use_actor_rnn:
                estimator_input_size = self.memory_a.rnn.hidden_size
                self.state_estimator = MlpModel(
                    input_size= estimator_input_size,
                    output_size= estimator_output_size,
                    **self.estimator_kwargs,
                )
            else:
                self.memory_s = Memory(
                    estimator_input_size,
                    type= kwargs.get("rnn_type", "lstm"),
                    num_layers= kwargs.get("rnn_num_layers", 1),
                    hidden_size= kwargs.get("rnn_hidden_size", 256),
                )
                self.state_estimator = MlpModel(
                    input_size= self.memory_s.rnn.hidden_size,
                    output_size= estimator_output_size,
                    **self.estimator_kwargs,
                )
        else:
            # Estimate required state using a feedforward network
            self.state_estimator = MlpModel(
                input_size= estimator_input_size,
                output_size= estimator_output_size,
                **self.estimator_kwargs,
            )
    
    def reset(self, dones=None):
        """Reset the estimator."""
        super().reset(dones)
        if self.is_recurrent and not self.use_actor_rnn:
            self.memory_s.reset(dones)

    def act(self, observations, masks=None, hidden_states=None, inference= False):
        """Perform an action based on the observations."""
        observations = observations.clone()
        if inference:
            assert masks is None and hidden_states is None, "Inference mode does not support masks and hidden_states."
        if self.is_recurrent and self.use_actor_rnn:
            input_s = self.memory_a(observations, masks, hidden_states)
            action = ActorCritic.act(self, input_s)
            self.estimated_state_ = self.state_estimator(input_s)
        elif self.is_recurrent and not self.use_actor_rnn:
            subobs = get_subobs_by_components(
                observations,
                component_names= self.estimator_obs_components,
                obs_segments= self.obs_segments,
            )
            input_s = self.memory_s(
                subobs,
                None, # use None to prevent unpadding
                hidden_states if hidden_states is None else hidden_states.estimator,
            )
            self.estimated_state_ = self.state_estimator(input_s)
            use_estimated_state_mask = torch.rand_like(observations[..., 0]) < self.replace_state_prob
            observations[use_estimated_state_mask] = substitute_estimated_state(
                observations[use_estimated_state_mask],
                self.estimator_target_components,
                self.estimated_state_[use_estimated_state_mask].detach(),
                self.obs_segments,
            )
            if inference:
                action = super().act_inference(observations)
            else:
                action = super().act(
                    observations,
                    masks= masks,
                    hidden_states= hidden_states if hidden_states is None else hidden_states.actor,
                )
        else:
            subobs = get_subobs_by_components(
                observations,
                component_names= self.estimator_obs_components,
                obs_segments= self.obs_segments,
            )
            self.estimated_state_ = self.state_estimator(subobs)
            use_estimated_state_mask = torch.rand_like(observations[..., 0]) < self.replace_state_prob
            observations[use_estimated_state_mask] = substitute_estimated_state(
                observations[use_estimated_state_mask],
                self.estimator_target_components,
                self.estimated_state_[use_estimated_state_mask].detach(),
                self.obs_segments,
            )
            if inference:
                action = super().act_inference(observations)
            else:
                action = super().act(observations, masks= masks, hidden_states= hidden_states)
        return action
    
    def act_inference(self, observations):
        """Perform an action in inference mode."""
        return self.act(observations, inference= True)
    
    def get_estimated_state(self):
        """Get the estimated state after calling act()."""
        return self.estimated_state_
    
    def get_hidden_states(self):
        """Get the hidden states."""
        return_ = super().get_hidden_states()
        if self.is_recurrent and not self.use_actor_rnn:
            return_ = return_._replace(actor= EstimatorActorHiddenState(
                self.memory_s.hidden_states,
                return_.actor,
            ))
        return return_
    
class StateAc(EstimatorMixin, ActorCritic):
    """State Actor-Critic class."""
    pass

class StateAcRecurrent(EstimatorMixin, ActorCriticRecurrent):
    """State Actor-Critic Recurrent class."""
    pass
