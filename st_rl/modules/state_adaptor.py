
import numpy as np

import torch
import torch.nn as nn
from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent, ActorCriticHiddenState
from st_rl.modules.mlp import MlpModel
from st_rl.modules.actor_critic_recurrent import Memory
from st_rl.modules.state_estimator import EstimatorMixin
from st_rl.utils import unpad_trajectories
from st_rl.utils.utils import get_subobs_size, get_subobs_by_components, substitute_estimated_state
from st_rl.utils.collections import namedarraytuple

AdaptorActorHiddenState = namedarraytuple('AdaptorActorHiddenState', [
    'adaptor',
    'actor',
])

class PrivilegeEstimatorMixin:
    def __init__(self,
            *args,
            adaptor_obs_components= None, # a list of strings used to get obs slices
            adaptor_target_components= None, # a list of strings used to get obs slices
            adaptor_kwargs= dict(),
            privilege_replace_state_prob= 0., # if 0~1, replace the actor observation with the estimated state with this probability
            **kwargs,
        ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.adaptor_obs_components = adaptor_obs_components
        self.adaptor_target_components = adaptor_target_components
        self.adaptor_kwargs = adaptor_kwargs
        self.privilege_replace_state_prob = privilege_replace_state_prob
        self.build_adaptor(**kwargs)
        print(f"Estimator: {self.state_adaptor}")

    def build_adaptor(self, **kwargs):
        """ This implementation is not flexible enough, but it is enough for now. """
        adaptor_input_size = get_subobs_size(self.obs_segments, self.adaptor_obs_components)
        adaptor_output_size = get_subobs_size(self.obs_segments, self.adaptor_target_components)
        if self.is_recurrent:
            self.adaptor_memory = Memory(
                    adaptor_input_size,
                    type= kwargs.get("rnn_type", "lstm"),
                    num_layers= kwargs.get("rnn_num_layers", 1),
                    hidden_size= kwargs.get("rnn_hidden_size", 256),
            )
            self.state_adaptor = MlpModel(
                    input_size= self.adaptor_memory.rnn.hidden_size,
                    output_size= adaptor_output_size,
                    **self.adaptor_kwargs,
            )
        else:
            # estimate required state using a feedforward network
            self.state_adaptor = MlpModel(
                input_size= adaptor_input_size,
                output_size= adaptor_output_size,
                **self.adaptor_kwargs,
            )
    
    def reset(self, dones=None):
        super().reset(dones)
        if self.is_recurrent:
            self.adaptor_memory.reset(dones)

    def act(self, observations, masks=None, hidden_states=None, inference= False):
        observations = observations.clone()
        if inference:
            assert masks is None and hidden_states is None, "Inference mode does not support masks and hidden_states. "
        if self.is_recurrent:
            # TODO: allows non-recurrent state adaptor with recurrent actor
            subobs = get_subobs_by_components(
                observations,
                component_names= self.adaptor_obs_components,
                obs_segments= self.obs_segments,
            )
            input_s = self.adaptor_memory(
                subobs,
                None, # use None to prevent unpadding
                hidden_states if hidden_states is None else hidden_states.adaptor,
            )
            # QUESTION: after memory_s, the estimated_state is already unpadded. How to get
            # the padded format and feed it into actor's observation?
            # SOLUTION: modify the code of Memory module, use masks= None to stop the unpadding.
            self.adaptor_state_ = self.state_adaptor(input_s)
            use_adaptor_state_mask = torch.rand_like(observations[..., 0]) < self.replace_state_prob
            observations[use_estimated_state_mask] = substitute_estimated_state(
                observations[use_estimated_state_mask],
                self.adaptor_target_components,
                self.adaptor_state_[use_adaptor_state_mask].detach(),
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
            # both state adaptor and actor are feedforward (non-recurrent)
            subobs = get_subobs_by_components(
                observations,
                component_names= self.adaptor_obs_components,
                obs_segments= self.obs_segments,
            )
            self.adaptor_state_ = self.state_adaptor(subobs)
            use_adaptor_state_mask = torch.rand_like(observations[..., 0]) < self.replace_state_prob
            observations[use_adaptor_state_mask] = substitute_estimated_state(
                observations[use_adaptor_state_mask],
                self.adaptor_target_components,
                self.adaptor_state_[use_adaptor_state_mask].detach(),
                self.obs_segments,
            )
            if inference:
                action = super().act_inference(observations)
            else:
                action = super().act(observations, masks= masks, hidden_states= hidden_states)
        return action
    
    def act_inference(self, observations):
        return self.act(observations, inference= True)
    
    """ No modification required for evaluate() """

    def get_estimated_state(self):
        """ In order to maintain the same interface of ActorCritic(Recurrent),
        the user must call this function after calling act() to get the estimated state.
        """
        return self.estimated_state_
    
    def get_hidden_states(self):
        return_ = super().get_hidden_states()
        if self.is_recurrent and not self.use_actor_rnn:
            return_ = return_._replace(actor= EstimatorActorHiddenState(
                self.memory_s.hidden_states,
                return_.actor,
            ))
        return return_
    

class PrivilegeStateAcRecurrent(PrivilegeEstimatorMixin, EstimatorMixin, ActorCriticRecurrent):
    pass
