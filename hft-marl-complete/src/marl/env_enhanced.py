"""
Enhanced CTDE Multi-Agent Environment for High-Frequency Trading
===============================================================

This module implements a realistic high-frequency trading environment with:
- Realistic market microstructure simulation
- Advanced risk management
- Multiple agent types (market makers, takers)
- Sophisticated reward functions
- Proper market impact modeling
- Latency and execution modeling

Based on the thesis: "Multi-agent deep reinforcement learning for high frequency trading"
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from enum import Enum


class AgentType(Enum):
    MARKET_MAKER = "market_maker"
    MARKET_TAKER = "market_taker"
    HYBRID = "hybrid"


@dataclass
class MarketConfig:
    """Configuration for market simulation parameters"""
    tick_size: float = 0.01
    decision_ms: int = 100
    max_spread_ticks: int = 10
    min_spread_ticks: int = 1
    volatility: float = 0.02
    mean_reversion_speed: float = 0.05
    liquidity_decay: float = 0.95
    impact_linear: float = 0.001
    impact_sqrt: float = 0.0005
    latency_ms: int = 1
    queue_position_noise: float = 0.1


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_inventory: int = 1000
    max_drawdown: float = 10000.0
    max_position_value: float = 50000.0
    kill_switch_drawdown: float = -5000.0
    max_order_rate_per_sec: float = 10.0
    max_pov: float = 0.05  # Percentage of Volume


@dataclass
class RewardConfig:
    """Reward function configuration"""
    lambda_inventory: float = 0.5
    lambda_impact: float = 0.2
    lambda_spread: float = 0.1
    lambda_risk: float = 1.0
    lambda_competition: float = 0.3
    profit_weight: float = 1.0
    inventory_penalty_power: float = 2.0


class MarketMicrostructure:
    """Advanced market microstructure simulation"""
    
    def __init__(self, config: MarketConfig):
        self.config = config
        self.order_book = {
            'bid': [],  # List of (price, quantity, agent_id)
            'ask': [],  # List of (price, quantity, agent_id)
        }
        self.mid_price = 100.0
        self.last_trade_price = 100.0
        self.volume_profile = {'bid': 0.0, 'ask': 0.0}
        self.latency_model = self._create_latency_model()
        
    def _create_latency_model(self) -> Dict[str, float]:
        """Create latency model for different order types"""
        return {
            'market': 0.5,  # ms
            'limit': 1.0,   # ms
            'cancel': 0.3,  # ms
            'modify': 0.8,  # ms
        }
    
    def get_best_bid_ask(self) -> Tuple[float, float, float, float]:
        """Get best bid/ask prices and quantities"""
        if not self.order_book['bid']:
            best_bid, bid_qty = 0.0, 0.0
        else:
            best_bid = max(self.order_book['bid'], key=lambda x: x[0])[0]
            bid_qty = sum(qty for price, qty, _ in self.order_book['bid'] if price == best_bid)
            
        if not self.order_book['ask']:
            best_ask, ask_qty = float('inf'), 0.0
        else:
            best_ask = min(self.order_book['ask'], key=lambda x: x[0])[0]
            ask_qty = sum(qty for price, qty, _ in self.order_book['ask'] if price == best_ask)
            
        return best_bid, best_ask, bid_qty, ask_qty
    
    def calculate_spread_imbalance(self) -> Tuple[float, float]:
        """Calculate spread and order book imbalance"""
        best_bid, best_ask, bid_qty, ask_qty = self.get_best_bid_ask()
        spread = best_ask - best_bid if best_ask != float('inf') else 0.0
        total_qty = bid_qty + ask_qty
        imbalance = (bid_qty - ask_qty) / total_qty if total_qty > 0 else 0.0
        return spread, imbalance
    
    def simulate_market_impact(self, signed_volume: float, is_market_order: bool = True) -> float:
        """Simulate market impact based on volume and order type"""
        if is_market_order:
            # Temporary impact for market orders
            linear_impact = self.config.impact_linear * abs(signed_volume)
            sqrt_impact = self.config.impact_sqrt * np.sqrt(abs(signed_volume))
            return signed_volume * (linear_impact + sqrt_impact)
        else:
            # Permanent impact for limit orders
            return signed_volume * self.config.impact_linear * 0.1
    
    def update_order_book(self, orders: List[Dict]) -> Dict[str, Any]:
        """Update order book with new orders and return execution results"""
        executions = []
        
        for order in orders:
            order_type = order['type']  # 'limit', 'market', 'cancel'
            side = order['side']  # 'buy', 'sell'
            quantity = order['quantity']
            price = order.get('price', 0.0)
            agent_id = order['agent_id']
            
            if order_type == 'limit':
                self._add_limit_order(side, price, quantity, agent_id)
            elif order_type == 'market':
                exec_result = self._execute_market_order(side, quantity, agent_id)
                executions.append(exec_result)
            elif order_type == 'cancel':
                self._cancel_order(agent_id, side, price)
        
        # Update mid price based on executions and market impact
        if executions:
            total_impact = sum(exec['impact'] for exec in executions)
            self.mid_price += total_impact
            
        return {'executions': executions, 'mid_price': self.mid_price}
    
    def _add_limit_order(self, side: str, price: float, quantity: float, agent_id: str):
        """Add a limit order to the book"""
        if side == 'buy':
            self.order_book['bid'].append((price, quantity, agent_id))
            self.order_book['bid'].sort(key=lambda x: -x[0])  # Sort by price desc
        else:
            self.order_book['ask'].append((price, quantity, agent_id))
            self.order_book['ask'].sort(key=lambda x: x[0])  # Sort by price asc
    
    def _execute_market_order(self, side: str, quantity: float, agent_id: str) -> Dict:
        """Execute a market order against the book"""
        executed_qty = 0.0
        executed_price = 0.0
        remaining_qty = quantity
        
        if side == 'buy':
            # Buy against ask orders
            while remaining_qty > 0 and self.order_book['ask']:
                best_ask_order = self.order_book['ask'][0]
                ask_price, ask_qty, _ = best_ask_order
                
                exec_qty = min(remaining_qty, ask_qty)
                executed_qty += exec_qty
                executed_price = ask_price
                
                # Update or remove the ask order
                if exec_qty >= ask_qty:
                    self.order_book['ask'].pop(0)
                else:
                    self.order_book['ask'][0] = (ask_price, ask_qty - exec_qty, best_ask_order[2])
                
                remaining_qty -= exec_qty
        else:
            # Sell against bid orders
            while remaining_qty > 0 and self.order_book['bid']:
                best_bid_order = self.order_book['bid'][0]
                bid_price, bid_qty, _ = best_bid_order
                
                exec_qty = min(remaining_qty, bid_qty)
                executed_qty += exec_qty
                executed_price = bid_price
                
                # Update or remove the bid order
                if exec_qty >= bid_qty:
                    self.order_book['bid'].pop(0)
                else:
                    self.order_book['bid'][0] = (bid_price, bid_qty - exec_qty, best_bid_order[2])
                
                remaining_qty -= exec_qty
        
        # Calculate market impact
        signed_volume = executed_qty if side == 'buy' else -executed_qty
        impact = self.simulate_market_impact(signed_volume, is_market_order=True)
        
        return {
            'agent_id': agent_id,
            'side': side,
            'quantity': executed_qty,
            'price': executed_price,
            'impact': impact,
            'remaining': remaining_qty
        }
    
    def _cancel_order(self, agent_id: str, side: str, price: float):
        """Cancel orders for a specific agent"""
        if side == 'buy':
            self.order_book['bid'] = [(p, q, aid) for p, q, aid in self.order_book['bid'] 
                                    if not (aid == agent_id and p == price)]
        else:
            self.order_book['ask'] = [(p, q, aid) for p, q, aid in self.order_book['ask'] 
                                    if not (aid == agent_id and p == price)]


class EnhancedCTDEHFTEnv(gym.Env):
    """
    Enhanced Centralized Training, Decentralized Execution environment
    for High-Frequency Trading with realistic market microstructure
    """
    
    def __init__(
        self,
        dataset_path: str,
        scaler_path: str,
        market_config: MarketConfig = None,
        risk_config: RiskConfig = None,
        reward_config: RewardConfig = None,
        episode_len: int = 1000,
        seed: int = 123,
        agent_types: List[AgentType] = None
    ):
        super().__init__()
        
        self.rng = np.random.default_rng(seed)
        self.market_config = market_config or MarketConfig()
        self.risk_config = risk_config or RiskConfig()
        self.reward_config = reward_config or RewardConfig()
        
        # Load dataset
        self.dataset = np.load(dataset_path)
        self.X = self.dataset['X']  # [N, T, F], scaled features
        self.N, self.T, self.F = self.X.shape
        
        # Agent setup
        self.agent_types = agent_types or [AgentType.MARKET_MAKER, AgentType.MARKET_TAKER]
        self.agents = [f"{agent_type.value}_{i}" for i, agent_type in enumerate(self.agent_types)]
        
        # Action and observation spaces
        self.action_space = gym.spaces.Dict({
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for agent in self.agents
        })
        
        # Enhanced observation space: features + agent state + market state
        obs_dim = self.F + 10  # features + agent_state + market_state
        self.observation_space = gym.spaces.Dict({
            agent: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        })
        
        # Episode configuration
        self.episode_len = min(episode_len, self.N - 1) if self.N > 1 else 0
        
        # Market microstructure
        self.market = MarketMicrostructure(self.market_config)
        
        # Agent states
        self.agent_states = {}
        self.episode_metrics = {}
        
        self.reset_state()
    
    def reset_state(self):
        """Reset environment state"""
        self.t = 0
        self.idx0 = int(self.rng.integers(low=0, high=max(1, self.N - self.episode_len)))
        
        # Initialize agent states
        for agent in self.agents:
            self.agent_states[agent] = {
                'inventory': 0.0,
                'cash': 0.0,
                'pnl': 0.0,
                'orders': [],
                'risk_limits_hit': [],
                'last_action': np.zeros(4),
                'action_history': [],
                'execution_history': []
            }
        
        # Initialize market
        self.market.mid_price = 100.0
        self.market.last_trade_price = 100.0
        self.market.order_book = {'bid': [], 'ask': []}
        
        # Episode metrics
        self.episode_metrics = {
            'total_volume': 0.0,
            'total_trades': 0,
            'market_impact': 0.0,
            'spread_evolution': [],
            'inventory_evolution': {agent: [] for agent in self.agents}
        }
    
    def _get_agent_observation(self, agent: str, frame: np.ndarray) -> np.ndarray:
        """Generate observation for a specific agent"""
        agent_state = self.agent_states[agent]
        
        # Market state
        spread, imbalance = self.market.calculate_spread_imbalance()
        best_bid, best_ask, bid_qty, ask_qty = self.market.get_best_bid_ask()
        
        market_state = np.array([
            spread,
            imbalance,
            self.market.mid_price,
            best_bid,
            best_ask,
            bid_qty,
            ask_qty,
            agent_state['inventory'],
            agent_state['cash'],
            len(agent_state['orders'])
        ], dtype=np.float32)
        
        # Combine features with agent and market state
        obs = np.concatenate([frame, market_state], axis=0)
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.reset_state()
        frame = self.X[self.idx0 + self.t, -1, :]
        
        obs = {}
        for agent in self.agents:
            obs[agent] = self._get_agent_observation(agent, frame)
        
        return obs, {}
    
    def _parse_actions(self, actions: Dict[str, np.ndarray]) -> List[Dict]:
        """Parse agent actions into market orders"""
        orders = []
        
        for agent, action in actions.items():
            agent_type = agent.split('_')[0] + '_' + agent.split('_')[1]
            
            if agent_type == 'market_maker':
                orders.extend(self._parse_market_maker_action(agent, action))
            elif agent_type == 'market_taker':
                orders.extend(self._parse_market_taker_action(agent, action))
        
        return orders
    
    def _parse_market_maker_action(self, agent: str, action: np.ndarray) -> List[Dict]:
        """Parse market maker action into limit orders"""
        orders = []
        
        # Action format: [bid_offset, ask_offset, bid_quantity, ask_quantity]
        bid_offset, ask_offset, bid_qty, ask_qty = action
        
        # Convert to tick offsets
        bid_offset_ticks = int(np.clip(bid_offset * 5, -5, 5))  # Scale to [-5, 5] ticks
        ask_offset_ticks = int(np.clip(ask_offset * 5, -5, 5))
        
        # Calculate prices
        bid_price = self.market.mid_price - (self.market_config.tick_size * bid_offset_ticks)
        ask_price = self.market.mid_price + (self.market_config.tick_size * ask_offset_ticks)
        
        # Quantities (normalized to reasonable sizes)
        bid_quantity = max(1, int(bid_qty * 10))
        ask_quantity = max(1, int(ask_qty * 10))
        
        # Risk checks
        if self._check_risk_limits(agent, 'bid', bid_quantity, bid_price):
            orders.append({
                'type': 'limit',
                'side': 'buy',
                'price': bid_price,
                'quantity': bid_quantity,
                'agent_id': agent
            })
        
        if self._check_risk_limits(agent, 'ask', ask_quantity, ask_price):
            orders.append({
                'type': 'limit',
                'side': 'sell',
                'price': ask_price,
                'quantity': ask_quantity,
                'agent_id': agent
            })
        
        return orders
    
    def _parse_market_taker_action(self, agent: str, action: np.ndarray) -> List[Dict]:
        """Parse market taker action into market orders"""
        orders = []
        
        # Action format: [direction, quantity, urgency, timing]
        direction, quantity, urgency, timing = action
        
        # Determine order side
        side = 'buy' if direction > 0 else 'sell'
        
        # Calculate quantity based on urgency and risk limits
        base_quantity = max(1, int(abs(quantity) * 20))
        final_quantity = int(base_quantity * (0.5 + 0.5 * urgency))
        
        # Risk checks
        if self._check_risk_limits(agent, side, final_quantity, 0):
            orders.append({
                'type': 'market',
                'side': side,
                'quantity': final_quantity,
                'agent_id': agent
            })
        
        return orders
    
    def _check_risk_limits(self, agent: str, side: str, quantity: float, price: float) -> bool:
        """Check if order violates risk limits"""
        agent_state = self.agent_states[agent]
        
        # Inventory limits
        if side == 'buy':
            new_inventory = agent_state['inventory'] + quantity
        else:
            new_inventory = agent_state['inventory'] - quantity
        
        if abs(new_inventory) > self.risk_config.max_inventory:
            return False
        
        # Position value limits
        position_value = abs(new_inventory) * self.market.mid_price
        if position_value > self.risk_config.max_position_value:
            return False
        
        # Drawdown limits
        if agent_state['pnl'] < self.risk_config.kill_switch_drawdown:
            return False
        
        return True
    
    def _calculate_reward(self, agent: str, execution_result: Dict) -> float:
        """Calculate reward for an agent"""
        agent_state = self.agent_states[agent]
        
        # Base profit from execution
        if execution_result['quantity'] > 0:
            if execution_result['side'] == 'buy':
                profit = -execution_result['price'] * execution_result['quantity']
                agent_state['inventory'] += execution_result['quantity']
                agent_state['cash'] += profit
            else:
                profit = execution_result['price'] * execution_result['quantity']
                agent_state['inventory'] -= execution_result['quantity']
                agent_state['cash'] += profit
            
            agent_state['pnl'] = agent_state['cash'] + agent_state['inventory'] * self.market.mid_price
        
        # Reward components
        profit_component = agent_state['pnl'] * self.reward_config.profit_weight
        
        # Inventory penalty (quadratic)
        inventory_penalty = self.reward_config.lambda_inventory * (
            abs(agent_state['inventory']) ** self.reward_config.inventory_penalty_power
        )
        
        # Impact penalty
        impact_penalty = self.reward_config.lambda_impact * abs(execution_result.get('impact', 0))
        
        # Risk penalty
        risk_penalty = 0.0
        if abs(agent_state['inventory']) > self.risk_config.max_inventory * 0.8:
            risk_penalty = self.reward_config.lambda_risk * 10.0
        
        # Competition penalty (based on spread)
        spread, _ = self.market.calculate_spread_imbalance()
        competition_penalty = self.reward_config.lambda_competition * spread
        
        total_reward = profit_component - inventory_penalty - impact_penalty - risk_penalty - competition_penalty
        
        return float(total_reward)
    
    def step(self, actions: Dict[str, np.ndarray]):
        """Execute one step in the environment"""
        # Parse actions into market orders
        orders = self._parse_actions(actions)
        
        # Update order book and get executions
        market_result = self.market.update_order_book(orders)
        executions = market_result['executions']
        
        # Update agent states and calculate rewards
        rewards = {}
        for agent in self.agents:
            # Find executions for this agent
            agent_executions = [exec for exec in executions if exec['agent_id'] == agent]
            
            if agent_executions:
                # Calculate reward based on executions
                total_reward = 0.0
                for execution in agent_executions:
                    reward = self._calculate_reward(agent, execution)
                    total_reward += reward
                    self.agent_states[agent]['execution_history'].append(execution)
                
                rewards[agent] = total_reward
            else:
                # No execution - small penalty for inventory holding
                inventory_penalty = self.reward_config.lambda_inventory * abs(
                    self.agent_states[agent]['inventory']
                ) * 0.1
                rewards[agent] = -inventory_penalty
        
        # Update episode metrics
        self.episode_metrics['total_volume'] += sum(exec['quantity'] for exec in executions)
        self.episode_metrics['total_trades'] += len(executions)
        self.episode_metrics['market_impact'] += sum(abs(exec['impact']) for exec in executions)
        
        spread, _ = self.market.calculate_spread_imbalance()
        self.episode_metrics['spread_evolution'].append(spread)
        
        for agent in self.agents:
            self.episode_metrics['inventory_evolution'][agent].append(
                self.agent_states[agent]['inventory']
            )
        
        # Generate next observations
        self.t += 1
        terminated = self.t >= self.episode_len
        
        if not terminated:
            frame = self.X[self.idx0 + self.t, -1, :]
            obs = {}
            for agent in self.agents:
                obs[agent] = self._get_agent_observation(agent, frame)
        else:
            obs = {agent: np.zeros(self.observation_space[agent].shape[0]) for agent in self.agents}
        
        # Termination and truncation
        term = {agent: terminated for agent in self.agents}
        trunc = {agent: False for agent in self.agents}
        
        # Additional info
        infos = {}
        for agent in self.agents:
            infos[agent] = {
                'inventory': self.agent_states[agent]['inventory'],
                'cash': self.agent_states[agent]['cash'],
                'pnl': self.agent_states[agent]['pnl'],
                'num_executions': len(self.agent_states[agent]['execution_history']),
                'spread': spread,
                'mid_price': self.market.mid_price
            }
        
        return obs, rewards, term, trunc, infos
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get comprehensive episode metrics"""
        return {
            'episode_length': self.t,
            'total_volume': self.episode_metrics['total_volume'],
            'total_trades': self.episode_metrics['total_trades'],
            'market_impact': self.episode_metrics['market_impact'],
            'final_spread': self.episode_metrics['spread_evolution'][-1] if self.episode_metrics['spread_evolution'] else 0.0,
            'agent_metrics': {
                agent: {
                    'final_inventory': state['inventory'],
                    'final_cash': state['cash'],
                    'final_pnl': state['pnl'],
                    'num_executions': len(state['execution_history'])
                }
                for agent, state in self.agent_states.items()
            }
        }
