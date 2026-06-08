import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import config


class Actor(nn.Module):
    """
    Policy function for PPO
    """
    def __init__(self,) -> None:
        super().__init__()
        # actual dense model
        # takes in the state and returns the action distribution (mean/logvar) to sample from
        self.net: nn.Module = nn.Sequential(
            nn.Linear(14, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 12),
        )
        
        
    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the action distribution for the policy

        Args:
            obs (torch.Tensor): Observations

        Returns:
            torch.Tensor: Distribution parameters
        """
        out = self.net(obs,)
        # Split last dimension into mean/logstd
        # since model can output negative values, we operate in logspace
        # that way, we can exponentiate and have a functional variance
        return (
            out[..., 0:8],
            torch.exp(
                out[..., 8:16]
            ),
        )
        
    
    @classmethod
    def gae(
        cls,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        value_outs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes GAE-derived advantages

        Args:
            rewards (torch.Tensor): Rewards at every step
            dones (torch.Tensor): Episode termination flags at every step
            value_outs (torch.Tensor): Predicted values of the policy

        Returns:
            torch.Tensor: GAE advantages
        """
        # gae is complex, but it effectively merges two distinct advantage computations (advantage being how much better/worse the policy performed than the critic predicted)
        # Compute TD residuals
        # td residuals "bootstrap" later rewards by discounting the value of the next state (which is the value given an optimal policy)
        # this is low variance (naturally the policy is a continuous smooth function and similar states yield similar distributions)
        td_residuals: torch.Tensor = rewards + config["rl"]["ppo"]["discount_factor"] * (1 - dones) * value_outs[1:, ...] - value_outs[:-1, ...]
        # Compute advantages
        # combine with the monte carlo advantages (direct computed discounted value based on all rewards of the states)
        advantages: torch.Tensor = torch.zeros_like(rewards) # T+1
        for t in reversed(range(td_residuals.size(0) - 1)):
            advantages[t, ...] = td_residuals[t, ...] + config["rl"]["ppo"]["discount_factor"] * config["rl"]["ppo"]["gae_decay"] * (1 - dones[t, ...]) * advantages[t, ...]
        
        return advantages
    
    
    @classmethod
    def policy_objective(
        cls,
        policy_dist: torch.distributions.Normal,
        old_policy_dist: torch.distributions.Normal,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the clipped surrogate objective for PPO

        Args:
            policy_dist (torch.distributions.Normal): Output distribution
            old_policy_dist (torch.distributions.Normal): Output distribution of the target policy
            actions (torch.Tensor): Sampled actions
            advantages (torch.Tensor): GAE advantages

        Returns:
            tuple[torch.Tensor, torch.Tensor]: a tuple containing the distribution parameters
        """
        advantages = advantages.detach()
        # Compute policy ratio of selected action
        policy_ratio: torch.Tensor = torch.exp(
            torch.sum(policy_dist.log_prob(actions), dim=2) - torch.sum(old_policy_dist.log_prob(actions).detach(), dim=2),
        )
        # Apply ratio scaling
        # effectively takes the "lower estimate" of the policy's performance and updates the gradients accordingly
        # if the policy is performing better than expected, make sure it doesnt dive too far into unexplored territory and prevent model collapse
        # otherwise, if the policy performed worse, let it catch up to the frozen policy
        policy_objecive: torch.Tensor = torch.minimum(
            advantages * policy_ratio,
            advantages * torch.clip(
                policy_ratio,
                1 - 0.2,
                1 + 0.2,
            ),
        )
        return policy_objecive.mean()
    
    
class Critic(nn.Module):
    """
    Value function for PPO
    """
    def __init__(self,) -> None:
        super().__init__()
        # similar architecture to the policy, just outputting a scalar (value) given the state
        self.net: nn.Module = nn.Sequential(
            nn.Linear(16, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
        
    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the expected value for the policy

        Args:
            obs (torch.Tensor): Observations

        Returns:
            torch.Tensor: Expected value
        """
        out: torch.Tensor = self.net(obs,)
        # Split last dimension into mean/logstd
        return out.squeeze(1)
    
    
    @classmethod
    def value_loss(
        cls,
        value_outs: torch.Tensor,
        old_value_outs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the value loss for PPO

        Args:
            value_outs (torch.Tensor): Predicted value of the policy
            old_value_outs (torch.Tensor): Old predicted value
            advantages (torch.Tensor): GAE advantages

        Returns:
            torch.Tensor: Value loss
        """
        # value function loss is simple: compute the mse of of the value outputs to the actual value netted by the policy (effectively, though recall the advantage isn't outright monte carlo, so the TD term is included)
        return F.mse_loss(
            value_outs,
            (advantages + old_value_outs).detach(),
        )