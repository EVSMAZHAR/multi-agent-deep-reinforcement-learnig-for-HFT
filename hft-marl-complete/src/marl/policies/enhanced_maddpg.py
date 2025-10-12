"""
Enhanced Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
================================================================

This module implements a comprehensive MADDPG algorithm with:
- Advanced actor-critic architectures
- Sophisticated exploration strategies
- Target network updates with polyak averaging
- Experience replay with prioritized sampling
- Gradient clipping and regularization
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
import random
from copy import deepcopy


@dataclass
class MADDPGConfig:
    """Configuration for MADDPG algorithm"""
    # Learning rates
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    
    # Network architecture
    hidden_dims: List[int] = None
    activation: str = 'relu'
    dropout: float = 0.1
    
    # Training parameters
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 1000000
    update_frequency: int = 1
    num_updates: int = 1
    
    # Exploration
    noise_type: str = 'gaussian'  # 'gaussian', 'ou', 'gumbel'
    noise_scale: float = 0.1
    noise_decay: float = 0.9995
    noise_min: float = 0.01
    
    # Advanced features
    use_prioritized_replay: bool = True
    use_double_q: bool = True
    use_spectral_norm: bool = False
    use_layer_norm: bool = True
    gradient_clip: float = 1.0
    
    # Risk management
    risk_penalty_weight: float = 0.1
    inventory_penalty_weight: float = 0.05
    max_inventory: float = 1000.0


class LayerNormLinear(nn.Module):
    """Linear layer with Layer Normalization"""
    
    def __init__(self, in_features: int, out_features: int, activation: str = 'relu'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.activation = getattr(F, activation)
        
    def forward(self, x):
        return self.activation(self.layer_norm(self.linear(x)))


class AttentionLayer(nn.Module):
    """Attention mechanism for multi-agent coordination"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1)
        output = torch.matmul(attention, v)
        
        return output


class EnhancedActor(nn.Module):
    """Enhanced actor network with advanced architecture"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: MADDPGConfig,
        agent_id: str = None
    ):
        super().__init__()
        self.config = config
        self.agent_id = agent_id
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = obs_dim
        
        hidden_dims = config.hidden_dims or [256, 256, 128]
        
        for hidden_dim in hidden_dims:
            if config.use_layer_norm:
                layers.append(LayerNormLinear(input_dim, hidden_dim, config.activation))
            else:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(getattr(nn, config.activation.title())())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Bounded actions
        
        self.network = nn.Sequential(*layers)
        
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
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(obs)


class EnhancedCritic(nn.Module):
    """Enhanced critic network with multi-agent coordination"""
    
    def __init__(
        self,
        joint_obs_dim: int,
        joint_action_dim: int,
        config: MADDPGConfig,
        num_agents: int
    ):
        super().__init__()
        self.config = config
        self.num_agents = num_agents
        
        # State processing branch
        state_layers = []
        input_dim = joint_obs_dim
        
        state_hidden_dims = config.hidden_dims or [256, 256]
        
        for hidden_dim in state_hidden_dims:
            if config.use_layer_norm:
                state_layers.append(LayerNormLinear(input_dim, hidden_dim, config.activation))
            else:
                state_layers.append(nn.Linear(input_dim, hidden_dim))
                state_layers.append(getattr(nn, config.activation.title())())
                if config.dropout > 0:
                    state_layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        
        self.state_network = nn.Sequential(*state_layers)
        
        # Action processing branch
        action_layers = []
        input_dim = joint_action_dim
        
        for hidden_dim in state_hidden_dims:
            if config.use_layer_norm:
                action_layers.append(LayerNormLinear(input_dim, hidden_dim, config.activation))
            else:
                action_layers.append(nn.Linear(input_dim, hidden_dim))
                action_layers.append(getattr(nn, config.activation.title())())
                if config.dropout > 0:
                    action_layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        
        self.action_network = nn.Sequential(*action_layers)
        
        # Attention mechanism for coordination
        self.attention = AttentionLayer(input_dim)
        
        # Output layers
        self.q1 = nn.Linear(input_dim, 1)
        if config.use_double_q:
            self.q2 = nn.Linear(input_dim, 1)
        
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
    
    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Process state and actions
        state_features = self.state_network(joint_obs)
        action_features = self.action_network(joint_actions)
        
        # Combine features
        combined = state_features + action_features
        
        # Apply attention
        attended = self.attention(combined)
        
        # Compute Q-values
        q1 = self.q1(attended)
        
        if self.config.use_double_q:
            q2 = self.q2(attended)
            return torch.min(q1, q2)  # Double Q-learning
        
        return q1


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def add(self, experience: Dict[str, Any], td_error: float = 1.0):
        """Add experience with priority"""
        priority = (abs(td_error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, torch.FloatTensor(weights), torch.LongTensor(indices)
    
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Update priorities for sampled experiences"""
        for i, idx in enumerate(indices):
            priority = (abs(td_errors[i].item()) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class NoiseGenerator:
    """Advanced noise generation for exploration"""
    
    def __init__(self, action_dim: int, config: MADDPGConfig):
        self.action_dim = action_dim
        self.config = config
        self.noise_scale = config.noise_scale
        
        if config.noise_type == 'ou':
            self.ou_noise = self._init_ou_noise()
    
    def _init_ou_noise(self):
        """Initialize Ornstein-Uhlenbeck noise"""
        return np.zeros(self.action_dim)
    
    def generate_noise(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate noise based on configuration"""
        if self.config.noise_type == 'gaussian':
            return np.random.normal(0, self.noise_scale, size)
        elif self.config.noise_type == 'ou':
            return self._generate_ou_noise(size)
        elif self.config.noise_type == 'gumbel':
            return self._generate_gumbel_noise(size)
        else:
            raise ValueError(f"Unknown noise type: {self.config.noise_type}")
    
    def _generate_ou_noise(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate Ornstein-Uhlenbeck noise"""
        noise = np.zeros(size)
        for i in range(size[0]):
            self.ou_noise = (1 - 0.15) * self.ou_noise + 0.15 * np.random.normal(0, 1, self.action_dim)
            noise[i] = self.ou_noise
        return noise * self.noise_scale
    
    def _generate_gumbel_noise(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate Gumbel noise for discrete actions"""
        return -np.log(-np.log(np.random.uniform(0, 1, size)))
    
    def decay_noise(self):
        """Decay noise scale"""
        self.noise_scale = max(self.config.noise_min, self.noise_scale * self.config.noise_decay)


class EnhancedMADDPG:
    """Enhanced Multi-Agent Deep Deterministic Policy Gradient"""
    
    def __init__(
        self,
        obs_dims: Dict[str, int],
        action_dims: Dict[str, int],
        agent_ids: List[str],
        config: MADDPGConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.config = config
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        
        # Networks
        self.actors = {}
        self.target_actors = {}
        self.critics = {}
        self.target_critics = {}
        
        # Joint dimensions
        self.joint_obs_dim = sum(obs_dims[agent] for agent in agent_ids)
        self.joint_action_dim = sum(action_dims[agent] for agent in agent_ids)
        
        # Initialize networks for each agent
        for agent_id in agent_ids:
            # Actor networks
            self.actors[agent_id] = EnhancedActor(
                obs_dims[agent_id], action_dims[agent_id], config, agent_id
            ).to(device)
            self.target_actors[agent_id] = EnhancedActor(
                obs_dims[agent_id], action_dims[agent_id], config, agent_id
            ).to(device)
            
            # Critic networks
            self.critics[agent_id] = EnhancedCritic(
                self.joint_obs_dim, self.joint_action_dim, config, self.num_agents
            ).to(device)
            self.target_critics[agent_id] = EnhancedCritic(
                self.joint_obs_dim, self.joint_action_dim, config, self.num_agents
            ).to(device)
            
            # Initialize target networks
            self._hard_update(self.target_actors[agent_id], self.actors[agent_id])
            self._hard_update(self.target_critics[agent_id], self.critics[agent_id])
        
        # Optimizers
        self.actor_optimizers = {
            agent_id: torch.optim.Adam(self.actors[agent_id].parameters(), lr=config.lr_actor)
            for agent_id in agent_ids
        }
        self.critic_optimizers = {
            agent_id: torch.optim.Adam(self.critics[agent_id].parameters(), lr=config.lr_critic)
            for agent_id in agent_ids
        }
        
        # Experience replay
        if config.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size)
        else:
            self.replay_buffer = deque(maxlen=config.buffer_size)
        
        # Noise generators
        self.noise_generators = {
            agent_id: NoiseGenerator(action_dims[agent_id], config)
            for agent_id in agent_ids
        }
        
        # Training statistics
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'td_error': deque(maxlen=100),
            'q_value': deque(maxlen=100)
        }
    
    def select_actions(self, observations: Dict[str, torch.Tensor], explore: bool = True) -> Dict[str, torch.Tensor]:
        """Select actions for all agents"""
        actions = {}
        
        for agent_id in self.agent_ids:
            obs = observations[agent_id].to(self.device)
            
            with torch.no_grad():
                action = self.actors[agent_id](obs)
            
            if explore:
                # Add exploration noise
                noise = self.noise_generators[agent_id].generate_noise(action.shape)
                noise = torch.FloatTensor(noise).to(self.device)
                action = action + noise
            
            # Clip actions to valid range
            action = torch.clamp(action, -1.0, 1.0)
            actions[agent_id] = action
        
        return actions
    
    def store_experience(self, experience: Dict[str, Any]):
        """Store experience in replay buffer"""
        if self.config.use_prioritized_replay:
            # For prioritized replay, we'll update the priority during training
            self.replay_buffer.add(experience, td_error=1.0)
        else:
            self.replay_buffer.append(experience)
    
    def train(self, batch_size: int = None) -> Dict[str, float]:
        """Train the MADDPG agents"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Check if we have enough experiences
        buffer_size = len(self.replay_buffer)
        if buffer_size < batch_size:
            return {}
        
        # Sample batch
        if self.config.use_prioritized_replay:
            experiences, weights, indices = self.replay_buffer.sample(batch_size)
            weights = weights.to(self.device)
        else:
            experiences = random.sample(self.replay_buffer, batch_size)
            weights = torch.ones(batch_size).to(self.device)
            indices = None
        
        # Convert to tensors
        batch = self._process_batch(experiences)
        
        # Train each agent
        training_info = {}
        
        for agent_id in self.agent_ids:
            agent_info = self._train_agent(agent_id, batch, weights, indices)
            training_info.update({f"{agent_id}_{k}": v for k, v in agent_info.items()})
        
        # Update noise scales
        for noise_gen in self.noise_generators.values():
            noise_gen.decay_noise()
        
        return training_info
    
    def _process_batch(self, experiences: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process batch of experiences into tensors"""
        batch_size = len(experiences)
        
        # Initialize batch dictionaries
        batch = {
            'observations': {agent_id: [] for agent_id in self.agent_ids},
            'actions': {agent_id: [] for agent_id in self.agent_ids},
            'rewards': {agent_id: [] for agent_id in self.agent_ids},
            'next_observations': {agent_id: [] for agent_id in self.agent_ids},
            'dones': []
        }
        
        # Fill batch
        for exp in experiences:
            for agent_id in self.agent_ids:
                batch['observations'][agent_id].append(exp['observations'][agent_id])
                batch['actions'][agent_id].append(exp['actions'][agent_id])
                batch['rewards'][agent_id].append(exp['rewards'][agent_id])
                batch['next_observations'][agent_id].append(exp['next_observations'][agent_id])
            batch['dones'].append(exp['dones'])
        
        # Convert to tensors
        for agent_id in self.agent_ids:
            batch['observations'][agent_id] = torch.stack(batch['observations'][agent_id]).to(self.device)
            batch['actions'][agent_id] = torch.stack(batch['actions'][agent_id]).to(self.device)
            batch['rewards'][agent_id] = torch.stack(batch['rewards'][agent_id]).to(self.device)
            batch['next_observations'][agent_id] = torch.stack(batch['next_observations'][agent_id]).to(self.device)
        
        batch['dones'] = torch.stack(batch['dones']).to(self.device)
        
        return batch
    
    def _train_agent(self, agent_id: str, batch: Dict[str, torch.Tensor], weights: torch.Tensor, indices: torch.Tensor) -> Dict[str, float]:
        """Train a specific agent"""
        # Get joint observations and actions
        joint_obs = torch.cat([batch['observations'][aid] for aid in self.agent_ids], dim=-1)
        joint_actions = torch.cat([batch['actions'][aid] for aid in self.agent_ids], dim=-1)
        joint_next_obs = torch.cat([batch['next_observations'][aid] for aid in self.agent_ids], dim=-1)
        
        # Current Q-values
        current_q = self.critics[agent_id](joint_obs, joint_actions)
        
        # Target actions from target actors
        with torch.no_grad():
            target_actions = {}
            for aid in self.agent_ids:
                target_actions[aid] = self.target_actors[aid](batch['next_observations'][aid])
            
            joint_target_actions = torch.cat([target_actions[aid] for aid in self.agent_ids], dim=-1)
            
            # Target Q-values
            target_q = self.target_critics[agent_id](joint_next_obs, joint_target_actions)
            
            # Compute targets
            rewards = batch['rewards'][agent_id]
            dones = batch['dones'][:, self.agent_ids.index(agent_id)]
            targets = rewards + self.config.gamma * target_q.squeeze() * (1 - dones)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q.squeeze(), targets)
        weighted_critic_loss = (critic_loss * weights).mean()
        
        # Update critic
        self.critic_optimizers[agent_id].zero_grad()
        weighted_critic_loss.backward()
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.config.gradient_clip)
        self.critic_optimizers[agent_id].step()
        
        # Actor loss (policy gradient)
        with torch.no_grad():
            current_actions = {}
            for aid in self.agent_ids:
                if aid == agent_id:
                    current_actions[aid] = self.actors[aid](batch['observations'][aid])
                else:
                    current_actions[aid] = batch['actions'][aid]
            
            joint_current_actions = torch.cat([current_actions[aid] for aid in self.agent_ids], dim=-1)
        
        # Compute policy gradient
        policy_actions = self.actors[agent_id](batch['observations'][agent_id])
        
        # Create joint actions with policy actions
        joint_policy_actions = joint_actions.clone()
        start_idx = sum(len(self.actors[aid](batch['observations'][aid])[0]) for aid in self.agent_ids[:self.agent_ids.index(agent_id)])
        end_idx = start_idx + len(policy_actions[0])
        joint_policy_actions[:, start_idx:end_idx] = policy_actions
        
        actor_loss = -self.critics[agent_id](joint_obs, joint_policy_actions).mean()
        
        # Add regularization terms
        if self.config.risk_penalty_weight > 0:
            # Risk penalty based on action variance
            action_variance = torch.var(policy_actions, dim=0).mean()
            actor_loss += self.config.risk_penalty_weight * action_variance
        
        # Update actor
        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.config.gradient_clip)
        self.actor_optimizers[agent_id].step()
        
        # Update target networks
        self._soft_update(self.target_actors[agent_id], self.actors[agent_id])
        self._soft_update(self.target_critics[agent_id], self.critics[agent_id])
        
        # Update priorities if using prioritized replay
        if self.config.use_prioritized_replay and indices is not None:
            with torch.no_grad():
                td_errors = torch.abs(current_q.squeeze() - targets)
                self.replay_buffer.update_priorities(indices, td_errors)
        
        # Store training statistics
        self.training_stats['actor_loss'].append(actor_loss.item())
        self.training_stats['critic_loss'].append(critic_loss.item())
        self.training_stats['td_error'].append(torch.abs(current_q.squeeze() - targets).mean().item())
        self.training_stats['q_value'].append(current_q.mean().item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'q_value': current_q.mean().item()
        }
    
    def _soft_update(self, target_network: nn.Module, source_network: nn.Module):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                self.config.tau * source_param.data + (1.0 - self.config.tau) * target_param.data
            )
    
    def _hard_update(self, target_network: nn.Module, source_network: nn.Module):
        """Hard update target network parameters"""
        target_network.load_state_dict(source_network.state_dict())
    
    def save_models(self, filepath: str):
        """Save all models to file"""
        save_dict = {
            'actors': {agent_id: self.actors[agent_id].state_dict() for agent_id in self.agent_ids},
            'critics': {agent_id: self.critics[agent_id].state_dict() for agent_id in self.agent_ids},
            'target_actors': {agent_id: self.target_actors[agent_id].state_dict() for agent_id in self.agent_ids},
            'target_critics': {agent_id: self.target_critics[agent_id].state_dict() for agent_id in self.agent_ids},
            'config': self.config,
            'agent_ids': self.agent_ids
        }
        torch.save(save_dict, filepath)
    
    def load_models(self, filepath: str):
        """Load models from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for agent_id in self.agent_ids:
            self.actors[agent_id].load_state_dict(checkpoint['actors'][agent_id])
            self.critics[agent_id].load_state_dict(checkpoint['critics'][agent_id])
            self.target_actors[agent_id].load_state_dict(checkpoint['target_actors'][agent_id])
            self.target_critics[agent_id].load_state_dict(checkpoint['target_critics'][agent_id])
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'std_{key}'] = np.std(values)
        return stats
