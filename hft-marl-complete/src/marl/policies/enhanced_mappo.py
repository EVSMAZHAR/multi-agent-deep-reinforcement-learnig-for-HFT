"""
Enhanced Multi-Agent Proximal Policy Optimization (MAPPO)
=========================================================

This module implements a comprehensive MAPPO algorithm with:
- Advanced actor-critic architectures with shared parameters
- Generalized Advantage Estimation (GAE) with adaptive lambda
- Multiple value function estimators
- Advanced entropy regularization
- Gradient clipping and adaptive learning rates
- Multi-agent coordination mechanisms
- Risk-aware policy learning

Based on the thesis: "Multi-agent deep reinforcement learning for high frequency trading"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import math


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO algorithm"""
    # Learning rates
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    
    # Network architecture
    hidden_dims: List[int] = None
    activation: str = 'relu'
    dropout: float = 0.1
    
    # PPO parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Training parameters
    epochs: int = 4
    batch_size: int = 64
    max_grad_norm: float = 0.5
    
    # Advanced features
    use_adaptive_lr: bool = True
    use_layer_norm: bool = True
    use_attention: bool = False
    use_spectral_norm: bool = False
    
    # Risk management
    risk_penalty_weight: float = 0.1
    inventory_penalty_weight: float = 0.05
    max_inventory: float = 1000.0


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with learnable parameters"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.alpha = nn.Parameter(torch.ones(1))  # Adaptive scaling
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        
        normalized = (x - mean) / std
        return self.alpha * (self.weight * normalized + self.bias)


class MultiHeadAttention(nn.Module):
    """Multi-head attention for agent coordination"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(context)
        
        return output, attention


class EnhancedActor(nn.Module):
    """Enhanced actor network for MAPPO"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: MAPPOConfig,
        agent_id: str = None
    ):
        super().__init__()
        self.config = config
        self.agent_id = agent_id
        self.action_dim = action_dim
        
        # Build shared layers
        layers = []
        input_dim = obs_dim
        
        hidden_dims = config.hidden_dims or [256, 256, 128]
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if config.use_layer_norm:
                layers.append(AdaptiveLayerNorm(hidden_dim))
            
            layers.append(getattr(nn, config.activation.title())())
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (for individual agent value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply spectral normalization if requested
        if config.use_spectral_norm:
            self.apply(lambda m: nn.utils.spectral_norm(m) if isinstance(m, nn.Linear) else m)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning policy, value, and features"""
        features = self.shared_layers(obs)
        
        policy = self.policy_head(features)
        value = self.value_head(features)
        
        return policy, value.squeeze(-1), features
    
    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        policy, value, features = self.forward(obs)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(probs=policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value


class CentralizedCritic(nn.Module):
    """Centralized critic for multi-agent coordination"""
    
    def __init__(
        self,
        joint_obs_dim: int,
        num_agents: int,
        config: MAPPOConfig
    ):
        super().__init__()
        self.config = config
        self.num_agents = num_agents
        
        # State processing
        layers = []
        input_dim = joint_obs_dim
        
        hidden_dims = config.hidden_dims or [512, 256, 128]
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if config.use_layer_norm:
                layers.append(AdaptiveLayerNorm(hidden_dim))
            
            layers.append(getattr(nn, config.activation.title())())
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Attention mechanism for coordination
        if config.use_attention:
            self.attention = MultiHeadAttention(input_dim)
        
        # Value estimation heads
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Risk estimation head
        self.risk_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply spectral normalization if requested
        if config.use_spectral_norm:
            self.apply(lambda m: nn.utils.spectral_norm(m) if isinstance(m, nn.Linear) else m)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, joint_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning value and risk estimates"""
        features = self.shared_layers(joint_obs)
        
        # Apply attention if enabled
        if self.config.use_attention:
            # Reshape for attention (batch_size, num_agents, feature_dim)
            batch_size = features.size(0) // self.num_agents
            features = features.view(batch_size, self.num_agents, -1)
            attended_features, attention_weights = self.attention(features)
            features = attended_features.view(-1, features.size(-1))
        
        value = self.value_head(features)
        risk = self.risk_head(features)
        
        return value.squeeze(-1), risk.squeeze(-1)


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler for MAPPO"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, patience: int = 10, factor: float = 0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.best_loss = float('inf')
        self.wait = 0
    
    def step(self, loss: float):
        """Update learning rate based on loss"""
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.factor
                self.wait = 0


class GAEComputer:
    """Generalized Advantage Estimation computer"""
    
    def __init__(self, gamma: float, gae_lambda: float):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        batch_size = rewards.size(0)
        
        if next_value is None:
            next_value = torch.zeros(1)
        
        # Compute advantages
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns


class EnhancedMAPPO:
    """Enhanced Multi-Agent Proximal Policy Optimization"""
    
    def __init__(
        self,
        obs_dims: Dict[str, int],
        action_dims: Dict[str, int],
        agent_ids: List[str],
        config: MAPPOConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.config = config
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        
        # Networks
        self.actors = {
            agent_id: EnhancedActor(obs_dims[agent_id], action_dims[agent_id], config, agent_id)
            for agent_id in agent_ids
        }.to(device)
        
        joint_obs_dim = sum(obs_dims[agent_id] for agent_id in agent_ids)
        self.critic = CentralizedCritic(joint_obs_dim, self.num_agents, config).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actors.parameters(), lr=config.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic
        )
        
        # Adaptive learning rate schedulers
        if config.use_adaptive_lr:
            self.actor_scheduler = AdaptiveLearningRateScheduler(self.actor_optimizer)
            self.critic_scheduler = AdaptiveLearningRateScheduler(self.critic_optimizer)
        
        # GAE computer
        self.gae_computer = GAEComputer(config.gamma, config.gae_lambda)
        
        # Training statistics
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy_loss': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100),
            'value_loss': deque(maxlen=100)
        }
    
    def select_actions(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Select actions for all agents"""
        actions = {}
        log_probs = {}
        values = {}
        
        for agent_id in self.agent_ids:
            obs = observations[agent_id].to(self.device)
            
            with torch.no_grad():
                action, log_prob, value = self.actors[agent_id].get_action(obs)
            
            actions[agent_id] = action
            log_probs[agent_id] = log_prob
            values[agent_id] = value
        
        return actions, log_probs, values
    
    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """Evaluate actions and return log probs, values, and entropy"""
        log_probs = {}
        values = {}
        entropies = {}
        
        for agent_id in self.agent_ids:
            obs = observations[agent_id].to(self.device)
            action = actions[agent_id].to(self.device)
            
            policy, value, features = self.actors[agent_id].forward(obs)
            
            # Compute log probability
            dist = torch.distributions.Categorical(probs=policy)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            log_probs[agent_id] = log_prob
            values[agent_id] = value
            entropies[agent_id] = entropy
        
        # Centralized value estimation
        joint_obs = torch.cat([observations[agent_id] for agent_id in self.agent_ids], dim=-1).to(self.device)
        central_values, risk_estimates = self.critic(joint_obs)
        
        return log_probs, values, central_values, entropies, risk_estimates
    
    def train(
        self,
        observations: List[Dict[str, torch.Tensor]],
        actions: List[Dict[str, torch.Tensor]],
        rewards: List[Dict[str, torch.Tensor]],
        dones: List[Dict[str, torch.Tensor]],
        old_log_probs: List[Dict[str, torch.Tensor]],
        values: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Train MAPPO agents"""
        
        # Convert to tensors
        batch_size = len(observations)
        
        # Stack observations and actions
        stacked_obs = {}
        stacked_actions = {}
        stacked_rewards = {}
        stacked_dones = {}
        stacked_old_log_probs = {}
        stacked_values = {}
        
        for agent_id in self.agent_ids:
            stacked_obs[agent_id] = torch.stack([obs[agent_id] for obs in observations]).to(self.device)
            stacked_actions[agent_id] = torch.stack([act[agent_id] for act in actions]).to(self.device)
            stacked_rewards[agent_id] = torch.stack([rew[agent_id] for rew in rewards]).to(self.device)
            stacked_dones[agent_id] = torch.stack([done[agent_id] for done in dones]).to(self.device)
            stacked_old_log_probs[agent_id] = torch.stack([old_lp[agent_id] for old_lp in old_log_probs]).to(self.device)
            stacked_values[agent_id] = torch.stack([val[agent_id] for val in values]).to(self.device)
        
        # Compute joint observations for critic
        joint_obs = torch.cat([stacked_obs[agent_id] for agent_id in self.agent_ids], dim=-1)
        
        # Compute GAE advantages and returns
        team_rewards = torch.stack([stacked_rewards[agent_id] for agent_id in self.agent_ids]).mean(dim=0)
        team_dones = torch.stack([stacked_dones[agent_id] for agent_id in self.agent_ids]).max(dim=0)[0]
        
        with torch.no_grad():
            central_values, _ = self.critic(joint_obs)
        
        advantages, returns = self.gae_computer.compute_gae(
            team_rewards, central_values, team_dones
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        training_info = {}
        
        for epoch in range(self.config.epochs):
            # Create mini-batches
            batch_indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, batch_size)
                mini_batch_indices = batch_indices[start_idx:end_idx]
                
                # Get mini-batch data
                mini_batch_obs = {agent_id: stacked_obs[agent_id][mini_batch_indices] for agent_id in self.agent_ids}
                mini_batch_actions = {agent_id: stacked_actions[agent_id][mini_batch_indices] for agent_id in self.agent_ids}
                mini_batch_old_log_probs = {agent_id: stacked_old_log_probs[agent_id][mini_batch_indices] for agent_id in self.agent_ids}
                mini_batch_advantages = advantages[mini_batch_indices]
                mini_batch_returns = returns[mini_batch_indices]
                mini_batch_central_values = central_values[mini_batch_indices]
                
                # Update actors
                actor_loss, entropy_loss, kl_div = self._update_actors(
                    mini_batch_obs, mini_batch_actions, mini_batch_old_log_probs, mini_batch_advantages
                )
                
                # Update critic
                critic_loss = self._update_critic(
                    mini_batch_obs, mini_batch_returns, mini_batch_central_values
                )
                
                # Store training statistics
                self.training_stats['actor_loss'].append(actor_loss)
                self.training_stats['critic_loss'].append(critic_loss)
                self.training_stats['entropy_loss'].append(entropy_loss)
                self.training_stats['kl_divergence'].append(kl_div)
        
        # Update learning rates if adaptive
        if self.config.use_adaptive_lr:
            self.actor_scheduler.step(actor_loss)
            self.critic_scheduler.step(critic_loss)
        
        # Return training information
        training_info = {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss,
            'kl_divergence': kl_div
        }
        
        return training_info
    
    def _update_actors(
        self,
        observations: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        old_log_probs: Dict[str, torch.Tensor],
        advantages: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Update actor networks"""
        
        # Evaluate current policy
        log_probs, values, central_values, entropies, risk_estimates = self.evaluate_actions(
            observations, actions
        )
        
        # Compute policy loss for each agent
        actor_losses = []
        entropy_losses = []
        kl_divergences = []
        
        for agent_id in self.agent_ids:
            # Compute ratio
            ratio = torch.exp(log_probs[agent_id] - old_log_probs[agent_id])
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy loss
            entropy_loss = -entropies[agent_id].mean()
            
            # KL divergence
            kl_div = (old_log_probs[agent_id] - log_probs[agent_id]).mean()
            
            # Total loss
            total_loss = policy_loss + self.config.entropy_coef * entropy_loss
            
            # Add risk penalty
            if self.config.risk_penalty_weight > 0:
                risk_penalty = self.config.risk_penalty_weight * risk_estimates.mean()
                total_loss += risk_penalty
            
            actor_losses.append(total_loss)
            entropy_losses.append(entropy_loss)
            kl_divergences.append(kl_div)
        
        # Update actors
        self.actor_optimizer.zero_grad()
        total_actor_loss = torch.stack(actor_losses).sum()
        total_actor_loss.backward()
        
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actors.parameters(), self.config.max_grad_norm)
        
        self.actor_optimizer.step()
        
        return (
            total_actor_loss.item(),
            torch.stack(entropy_losses).mean().item(),
            torch.stack(kl_divergences).mean().item()
        )
    
    def _update_critic(
        self,
        observations: Dict[str, torch.Tensor],
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> float:
        """Update critic network"""
        
        # Compute joint observations
        joint_obs = torch.cat([observations[agent_id] for agent_id in self.agent_ids], dim=-1)
        
        # Get current value estimates
        current_values, risk_estimates = self.critic(joint_obs)
        
        # Compute value loss
        value_loss = F.mse_loss(current_values, returns)
        
        # Add risk estimation loss
        if self.config.risk_penalty_weight > 0:
            risk_target = torch.abs(returns - old_values)
            risk_loss = F.mse_loss(risk_estimates, risk_target)
            value_loss += self.config.risk_penalty_weight * risk_loss
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        
        self.critic_optimizer.step()
        
        return value_loss.item()
    
    def save_models(self, filepath: str):
        """Save all models to file"""
        save_dict = {
            'actors': self.actors.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
            'agent_ids': self.agent_ids
        }
        torch.save(save_dict, filepath)
    
    def load_models(self, filepath: str):
        """Load models from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actors.load_state_dict(checkpoint['actors'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'std_{key}'] = np.std(values)
        return stats
