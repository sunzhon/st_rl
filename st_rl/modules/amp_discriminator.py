import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

from st_rl.utils import utils
from .actor_critic import ActorCritic, get_activation
from .actor_critic_recurrent import ActorCriticRecurrent, get_activation


class AMPDiscriminator(nn.Module):
    def __init__(
            self, input_dim, amp_reward_coef, hidden_dims, task_reward_lerp=0.0, amp_grad_pen=5,device="cuda:0"):
        """
        Initialize the AMPDiscriminator.

        Args:
            input_dim: Dimension of the input.
            amp_reward_coef: Coefficient for AMP reward.
            hidden_dims: List of sizes for hidden layers.
            device: Device to run the model on (e.g., 'cpu' or 'cuda').
            task_reward_lerp: Linear interpolation factor for task reward.
        """
        super(AMPDiscriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        self.amp_grad_pen = amp_grad_pen
        self.task_reward_lerp = task_reward_lerp

        # Build the discriminator network
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_dims:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_dims[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x: Input tensor.

        Returns:
            Discriminator output.
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,amp_obs):
        """
        Compute the gradient penalty for the discriminator.

        Args:
            expert_state: Expert state tensor.
            lambda_: Gradient penalty coefficient.

        Returns:
            Gradient penalty.
        """
        expert_data = amp_obs.clone()
        expert_data.requires_grad = True
        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = self.amp_grad_pen * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        """
        Predict the AMP reward.

        Args:
            state: Current state tensor.
            next_state: Next state tensor.
            task_reward: Task reward tensor.
            normalizer: Normalizer for the state tensors.

        Returns:
            Predicted AMP reward and discriminator output.
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)
            
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            # if the amp_obs of policy was discrimated as that from expert obs, then should get high rewards
            reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze() #, d

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r: Discriminator reward tensor.
            task_r: Task reward tensor.

        Returns:
            Interpolated reward tensor.
        """
        return (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
    

class AmpMixin:
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        """
        Initialize the AmpActorCritic.

        Args:
            num_actor_obs: Number of actor observations.
            num_critic_obs: Number of critic observations.
            num_actions: Number of actions.
            actor_hidden_dims: List of sizes for actor hidden layers.
            critic_hidden_dims: List of sizes for critic hidden layers.
            activation: Activation function.
            rnn_type: Type of RNN (e.g., 'lstm').
            rnn_hidden_size: Size of RNN hidden layer.
            rnn_num_layers: Number of RNN layers.
            init_noise_std: Initial noise standard deviation.
        """
        super().__init__(num_actor_obs=num_actor_obs,
                         num_critic_obs=num_critic_obs,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         rnn_type=rnn_type,
                         rnn_hidden_size=rnn_hidden_size,
                         rnn_num_layers=rnn_num_layers,
                         init_noise_std=init_noise_std,
                         **kwargs,
                        )

        cfg = kwargs.get('amp_discriminator', {})
        self.discriminator = AMPDiscriminator(**cfg)
        print(f"Discriminator: {self.discriminator}")

class AmpActorCritic( AmpMixin, ActorCritic):
    pass

class AmpActorCriticRecurrent( AmpMixin, ActorCriticRecurrent):
    pass
