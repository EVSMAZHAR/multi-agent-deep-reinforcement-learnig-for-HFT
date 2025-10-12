"""
Comprehensive Baseline Algorithms for High-Frequency Trading
============================================================

This module implements a comprehensive set of baseline algorithms for comparison
with the multi-agent reinforcement learning approaches:

1. Avellaneda-Stoikov Market Making
2. TWAP (Time-Weighted Average Price)
3. VWAP (Volume-Weighted Average Price)
4. POV (Percentage of Volume)
5. Implementation Shortfall
6. Risk Parity
7. Momentum/Mean Reversion Strategies
8. Random Walk
9. Buy and Hold
10. Technical Analysis Based Strategies

Based on the thesis: "Multi-agent deep reinforcement learning for high frequency trading"
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import scipy.optimize
from scipy.stats import norm


class StrategyType(Enum):
    MARKET_MAKING = "market_making"
    EXECUTION = "execution"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TECHNICAL = "technical"
    BENCHMARK = "benchmark"


@dataclass
class MarketState:
    """Market state information"""
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    imbalance: float
    volume: float
    volatility: float
    timestamp: float


@dataclass
class Order:
    """Order representation"""
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str  # 'limit', 'market'
    timestamp: float


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, strategy_type: StrategyType):
        self.name = name
        self.strategy_type = strategy_type
        self.positions = {}
        self.trades = []
        self.pnl_history = []
        
    @abstractmethod
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate orders based on market state and agent state"""
        pass
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update strategy state after execution"""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the strategy"""
        if not self.pnl_history:
            return {}
        
        returns = np.array(self.pnl_history)
        metrics = {
            'total_return': returns[-1] - returns[0],
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': np.mean(np.diff(returns) > 0),
            'profit_factor': self._calculate_profit_factor(returns)
        }
        return metrics
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        return np.sum(positive_returns) / abs(np.sum(negative_returns))


class AvellanedaStoikovStrategy(BaseStrategy):
    """Enhanced Avellaneda-Stoikov Market Making Strategy"""
    
    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        sigma: float = 0.02,
        tick_size: float = 0.01,
        max_inventory: float = 1000.0,
        risk_aversion: float = 0.01
    ):
        super().__init__("Avellaneda-Stoikov", StrategyType.MARKET_MAKING)
        self.gamma = gamma  # Risk aversion parameter
        self.k = k  # Order arrival parameter
        self.sigma = sigma  # Volatility
        self.tick_size = tick_size
        self.max_inventory = max_inventory
        self.risk_aversion = risk_aversion
        
        # Dynamic parameters
        self.time_horizon = 60.0  # seconds
        self.current_time = 0.0
        self.inventory = 0.0
        
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate optimal bid and ask quotes"""
        orders = []
        
        # Update time
        self.current_time = market_state.timestamp
        
        # Calculate remaining time
        time_to_go = max(0.1, self.time_horizon - self.current_time)
        
        # Calculate reservation price
        reservation_price = market_state.mid_price - self.gamma * self.sigma**2 * time_to_go * self.inventory
        
        # Calculate optimal spread
        half_spread = self.gamma * self.sigma**2 * time_to_go / 2 + np.log(1 + self.gamma / self.k) / self.gamma
        
        # Calculate bid and ask prices
        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread
        
        # Round to tick size
        bid_price = np.round(bid_price / self.tick_size) * self.tick_size
        ask_price = np.round(ask_price / self.tick_size) * self.tick_size
        
        # Risk management: adjust quantities based on inventory
        base_quantity = 100.0
        inventory_penalty = abs(self.inventory) / self.max_inventory
        
        bid_quantity = base_quantity * (1.0 - inventory_penalty) if self.inventory < self.max_inventory else 0
        ask_quantity = base_quantity * (1.0 - inventory_penalty) if self.inventory > -self.max_inventory else 0
        
        # Generate orders
        if bid_quantity > 0 and bid_price > 0:
            orders.append(Order(
                side='buy',
                quantity=bid_quantity,
                price=bid_price,
                order_type='limit',
                timestamp=market_state.timestamp
            ))
        
        if ask_quantity > 0:
            orders.append(Order(
                side='sell',
                quantity=ask_quantity,
                price=ask_price,
                order_type='limit',
                timestamp=market_state.timestamp
            ))
        
        return orders
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update inventory after execution"""
        if execution_result['side'] == 'buy':
            self.inventory += execution_result['quantity']
        else:
            self.inventory -= execution_result['quantity']


class TWAPStrategy(BaseStrategy):
    """Time-Weighted Average Price Strategy"""
    
    def __init__(
        self,
        target_quantity: float,
        time_horizon: float,
        slice_size: float = None,
        aggressiveness: float = 0.5
    ):
        super().__init__("TWAP", StrategyType.EXECUTION)
        self.target_quantity = target_quantity
        self.time_horizon = time_horizon
        self.slice_size = slice_size or (target_quantity / 10)  # 10 slices by default
        self.aggressiveness = aggressiveness  # 0 = passive, 1 = aggressive
        
        self.remaining_quantity = target_quantity
        self.start_time = None
        self.executed_quantity = 0.0
        
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate TWAP orders"""
        orders = []
        
        if self.start_time is None:
            self.start_time = market_state.timestamp
        
        # Calculate time progress
        elapsed_time = market_state.timestamp - self.start_time
        time_progress = min(1.0, elapsed_time / self.time_horizon)
        
        # Calculate target quantity for this slice
        target_slice = self.target_quantity * time_progress
        quantity_to_execute = target_slice - self.executed_quantity
        
        if quantity_to_execute > 0:
            # Determine order type based on aggressiveness
            if self.aggressiveness > 0.7:
                # Market order
                orders.append(Order(
                    side='buy',
                    quantity=min(quantity_to_execute, self.slice_size),
                    price=market_state.best_ask,  # Market buy
                    order_type='market',
                    timestamp=market_state.timestamp
                ))
            else:
                # Limit order
                limit_price = market_state.mid_price * (1.0 - self.aggressiveness * 0.001)
                orders.append(Order(
                    side='buy',
                    quantity=min(quantity_to_execute, self.slice_size),
                    price=limit_price,
                    order_type='limit',
                    timestamp=market_state.timestamp
                ))
        
        return orders
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update executed quantity"""
        self.executed_quantity += execution_result['quantity']
        self.remaining_quantity -= execution_result['quantity']


class VWAPStrategy(BaseStrategy):
    """Volume-Weighted Average Price Strategy"""
    
    def __init__(
        self,
        target_quantity: float,
        time_horizon: float,
        volume_profile: np.ndarray = None
    ):
        super().__init__("VWAP", StrategyType.EXECUTION)
        self.target_quantity = target_quantity
        self.time_horizon = time_horizon
        self.volume_profile = volume_profile  # Expected volume distribution
        
        self.executed_quantity = 0.0
        self.start_time = None
        self.volume_so_far = 0.0
        
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate VWAP orders"""
        orders = []
        
        if self.start_time is None:
            self.start_time = market_state.timestamp
        
        # Calculate time progress
        elapsed_time = market_state.timestamp - self.start_time
        time_progress = min(1.0, elapsed_time / self.time_horizon)
        
        # Calculate target participation rate
        if self.volume_profile is not None:
            # Use provided volume profile
            target_participation = self.volume_profile[int(time_progress * len(self.volume_profile))]
        else:
            # Assume uniform volume distribution
            target_participation = self.target_quantity / self.time_horizon
        
        # Calculate quantity based on market volume
        market_volume = market_state.volume
        quantity_to_execute = min(
            target_participation * market_volume,
            self.target_quantity - self.executed_quantity
        )
        
        if quantity_to_execute > 0:
            # Use limit order to minimize market impact
            limit_price = market_state.mid_price * 0.9999
            orders.append(Order(
                side='buy',
                quantity=quantity_to_execute,
                price=limit_price,
                order_type='limit',
                timestamp=market_state.timestamp
            ))
        
        return orders
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update executed quantity and volume"""
        self.executed_quantity += execution_result['quantity']
        self.volume_so_far += execution_result['quantity']


class POVStrategy(BaseStrategy):
    """Percentage of Volume Strategy"""
    
    def __init__(
        self,
        target_quantity: float,
        participation_rate: float = 0.1,  # 10% of volume
        max_order_size: float = 1000.0
    ):
        super().__init__("POV", StrategyType.EXECUTION)
        self.target_quantity = target_quantity
        self.participation_rate = participation_rate
        self.max_order_size = max_order_size
        
        self.executed_quantity = 0.0
        
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate POV orders"""
        orders = []
        
        # Calculate target quantity based on market volume
        target_quantity = min(
            market_state.volume * self.participation_rate,
            self.max_order_size,
            self.target_quantity - self.executed_quantity
        )
        
        if target_quantity > 0:
            # Use limit order
            limit_price = market_state.mid_price * 0.9999
            orders.append(Order(
                side='buy',
                quantity=target_quantity,
                price=limit_price,
                order_type='limit',
                timestamp=market_state.timestamp
            ))
        
        return orders
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update executed quantity"""
        self.executed_quantity += execution_result['quantity']


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(
        self,
        lookback_window: int = 20,
        momentum_threshold: float = 0.001,
        position_size: float = 100.0
    ):
        super().__init__("Momentum", StrategyType.MOMENTUM)
        self.lookback_window = lookback_window
        self.momentum_threshold = momentum_threshold
        self.position_size = position_size
        
        self.price_history = []
        self.current_position = 0.0
        
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate momentum-based orders"""
        orders = []
        
        # Update price history
        self.price_history.append(market_state.mid_price)
        if len(self.price_history) > self.lookback_window:
            self.price_history.pop(0)
        
        if len(self.price_history) < self.lookback_window:
            return orders
        
        # Calculate momentum
        momentum = (market_state.mid_price - self.price_history[0]) / self.price_history[0]
        
        # Generate orders based on momentum
        if momentum > self.momentum_threshold and self.current_position < self.position_size:
            # Strong upward momentum - buy
            orders.append(Order(
                side='buy',
                quantity=self.position_size - self.current_position,
                price=market_state.best_ask,
                order_type='market',
                timestamp=market_state.timestamp
            ))
        elif momentum < -self.momentum_threshold and self.current_position > -self.position_size:
            # Strong downward momentum - sell
            orders.append(Order(
                side='sell',
                quantity=self.current_position + self.position_size,
                price=market_state.best_bid,
                order_type='market',
                timestamp=market_state.timestamp
            ))
        
        return orders
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update position"""
        if execution_result['side'] == 'buy':
            self.current_position += execution_result['quantity']
        else:
            self.current_position -= execution_result['quantity']


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(
        self,
        lookback_window: int = 20,
        reversion_threshold: float = 2.0,  # Standard deviations
        position_size: float = 100.0
    ):
        super().__init__("Mean Reversion", StrategyType.MEAN_REVERSION)
        self.lookback_window = lookback_window
        self.reversion_threshold = reversion_threshold
        self.position_size = position_size
        
        self.price_history = []
        self.current_position = 0.0
        
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate mean reversion orders"""
        orders = []
        
        # Update price history
        self.price_history.append(market_state.mid_price)
        if len(self.price_history) > self.lookback_window:
            self.price_history.pop(0)
        
        if len(self.price_history) < self.lookback_window:
            return orders
        
        # Calculate z-score
        prices = np.array(self.price_history)
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price == 0:
            return orders
        
        z_score = (market_state.mid_price - mean_price) / std_price
        
        # Generate orders based on z-score
        if z_score > self.reversion_threshold and self.current_position > -self.position_size:
            # Price too high - sell (expect reversion down)
            orders.append(Order(
                side='sell',
                quantity=min(self.position_size + self.current_position, self.position_size),
                price=market_state.best_bid,
                order_type='limit',
                timestamp=market_state.timestamp
            ))
        elif z_score < -self.reversion_threshold and self.current_position < self.position_size:
            # Price too low - buy (expect reversion up)
            orders.append(Order(
                side='buy',
                quantity=min(self.position_size - self.current_position, self.position_size),
                price=market_state.best_ask,
                order_type='limit',
                timestamp=market_state.timestamp
            ))
        
        return orders
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update position"""
        if execution_result['side'] == 'buy':
            self.current_position += execution_result['quantity']
        else:
            self.current_position -= execution_result['quantity']


class TechnicalAnalysisStrategy(BaseStrategy):
    """Technical analysis based strategy"""
    
    def __init__(
        self,
        sma_short: int = 10,
        sma_long: int = 30,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        position_size: float = 100.0
    ):
        super().__init__("Technical Analysis", StrategyType.TECHNICAL)
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.position_size = position_size
        
        self.price_history = []
        self.current_position = 0.0
        
    def _calculate_sma(self, prices: np.ndarray, window: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < window:
            return prices[-1]
        return np.mean(prices[-window:])
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate technical analysis based orders"""
        orders = []
        
        # Update price history
        self.price_history.append(market_state.mid_price)
        
        if len(self.price_history) < max(self.sma_long, self.rsi_period) + 1:
            return orders
        
        # Calculate indicators
        sma_short = self._calculate_sma(np.array(self.price_history), self.sma_short)
        sma_long = self._calculate_sma(np.array(self.price_history), self.sma_long)
        rsi = self._calculate_rsi(np.array(self.price_history), self.rsi_period)
        
        # Generate signals
        sma_signal = 1 if sma_short > sma_long else -1
        rsi_signal = 0
        if rsi < self.rsi_oversold:
            rsi_signal = 1  # Oversold - buy signal
        elif rsi > self.rsi_overbought:
            rsi_signal = -1  # Overbought - sell signal
        
        # Combine signals
        combined_signal = sma_signal + rsi_signal
        
        # Generate orders
        if combined_signal > 0 and self.current_position < self.position_size:
            # Buy signal
            orders.append(Order(
                side='buy',
                quantity=min(self.position_size - self.current_position, self.position_size),
                price=market_state.best_ask,
                order_type='limit',
                timestamp=market_state.timestamp
            ))
        elif combined_signal < 0 and self.current_position > -self.position_size:
            # Sell signal
            orders.append(Order(
                side='sell',
                quantity=min(self.current_position + self.position_size, self.position_size),
                price=market_state.best_bid,
                order_type='limit',
                timestamp=market_state.timestamp
            ))
        
        return orders
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update position"""
        if execution_result['side'] == 'buy':
            self.current_position += execution_result['quantity']
        else:
            self.current_position -= execution_result['quantity']


class RandomWalkStrategy(BaseStrategy):
    """Random walk strategy for baseline comparison"""
    
    def __init__(
        self,
        order_probability: float = 0.1,
        max_position_size: float = 100.0,
        order_size: float = 10.0
    ):
        super().__init__("Random Walk", StrategyType.BENCHMARK)
        self.order_probability = order_probability
        self.max_position_size = max_position_size
        self.order_size = order_size
        self.current_position = 0.0
        
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate random orders"""
        orders = []
        
        # Randomly decide whether to place an order
        if np.random.random() < self.order_probability:
            # Randomly choose side
            side = 'buy' if np.random.random() < 0.5 else 'sell'
            
            # Check position limits
            if side == 'buy' and self.current_position < self.max_position_size:
                orders.append(Order(
                    side=side,
                    quantity=self.order_size,
                    price=market_state.best_ask,
                    order_type='market',
                    timestamp=market_state.timestamp
                ))
            elif side == 'sell' and self.current_position > -self.max_position_size:
                orders.append(Order(
                    side=side,
                    quantity=self.order_size,
                    price=market_state.best_bid,
                    order_type='market',
                    timestamp=market_state.timestamp
                ))
        
        return orders
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update position"""
        if execution_result['side'] == 'buy':
            self.current_position += execution_result['quantity']
        else:
            self.current_position -= execution_result['quantity']


class BuyAndHoldStrategy(BaseStrategy):
    """Buy and hold strategy for benchmark comparison"""
    
    def __init__(self, target_quantity: float = 100.0):
        super().__init__("Buy and Hold", StrategyType.BENCHMARK)
        self.target_quantity = target_quantity
        self.executed = False
        
    def generate_orders(self, market_state: MarketState, agent_state: Dict[str, Any]) -> List[Order]:
        """Generate buy and hold order"""
        orders = []
        
        if not self.executed:
            orders.append(Order(
                side='buy',
                quantity=self.target_quantity,
                price=market_state.best_ask,
                order_type='market',
                timestamp=market_state.timestamp
            ))
            self.executed = True
        
        return orders
    
    def update_state(self, execution_result: Dict[str, Any]):
        """Update state"""
        pass


class BaselineManager:
    """Manager for all baseline strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.performance_history = {}
        
    def add_strategy(self, name: str, strategy: BaseStrategy):
        """Add a strategy to the manager"""
        self.strategies[name] = strategy
        self.performance_history[name] = []
    
    def get_strategy(self, name: str) -> BaseStrategy:
        """Get a strategy by name"""
        return self.strategies.get(name)
    
    def run_backtest(
        self,
        market_data: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> Dict[str, Dict[str, Any]]:
        """Run backtest for all strategies"""
        results = {}
        
        for name, strategy in self.strategies.items():
            # Reset strategy state
            strategy.positions = {}
            strategy.trades = []
            strategy.pnl_history = [initial_capital]
            
            # Run backtest
            strategy_results = self._run_single_backtest(strategy, market_data, initial_capital)
            results[name] = strategy_results
        
        return results
    
    def _run_single_backtest(
        self,
        strategy: BaseStrategy,
        market_data: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, Any]:
        """Run backtest for a single strategy"""
        current_capital = initial_capital
        current_position = 0.0
        
        for _, row in market_data.iterrows():
            # Create market state
            market_state = MarketState(
                mid_price=row['mid_price'],
                best_bid=row['best_bid'],
                best_ask=row['best_ask'],
                spread=row['best_ask'] - row['best_bid'],
                imbalance=row.get('imbalance', 0.0),
                volume=row.get('volume', 100.0),
                volatility=row.get('volatility', 0.02),
                timestamp=row['timestamp']
            )
            
            # Create agent state
            agent_state = {
                'capital': current_capital,
                'position': current_position,
                'pnl': current_capital + current_position * market_state.mid_price - initial_capital
            }
            
            # Generate orders
            orders = strategy.generate_orders(market_state, agent_state)
            
            # Simulate order execution (simplified)
            for order in orders:
                if order.order_type == 'market':
                    # Market order executes immediately
                    execution_price = order.price
                    execution_quantity = order.quantity
                    
                    # Update capital and position
                    if order.side == 'buy':
                        current_capital -= execution_price * execution_quantity
                        current_position += execution_quantity
                    else:
                        current_capital += execution_price * execution_quantity
                        current_position -= execution_quantity
                    
                    # Update strategy state
                    strategy.update_state({
                        'side': order.side,
                        'quantity': execution_quantity,
                        'price': execution_price
                    })
            
            # Update PnL history
            current_pnl = current_capital + current_position * market_state.mid_price - initial_capital
            strategy.pnl_history.append(current_pnl)
        
        # Calculate final performance metrics
        performance_metrics = strategy.get_performance_metrics()
        
        return {
            'final_capital': current_capital,
            'final_position': current_position,
            'total_trades': len(strategy.trades),
            'performance_metrics': performance_metrics,
            'pnl_history': strategy.pnl_history
        }
    
    def compare_strategies(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare performance of all strategies"""
        comparison_data = []
        
        for strategy_name, result in results.items():
            metrics = result['performance_metrics']
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': metrics.get('total_return', 0.0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0.0),
                'Max Drawdown': metrics.get('max_drawdown', 0.0),
                'Win Rate': metrics.get('win_rate', 0.0),
                'Profit Factor': metrics.get('profit_factor', 0.0),
                'Total Trades': result['total_trades'],
                'Final Capital': result['final_capital']
            })
        
        return pd.DataFrame(comparison_data)


# Factory function for creating baseline strategies
def create_baseline_strategies() -> BaselineManager:
    """Create a comprehensive set of baseline strategies"""
    manager = BaselineManager()
    
    # Market making strategies
    manager.add_strategy("Avellaneda-Stoikov", AvellanedaStoikovStrategy())
    
    # Execution strategies
    manager.add_strategy("TWAP", TWAPStrategy(target_quantity=1000.0, time_horizon=3600.0))
    manager.add_strategy("VWAP", VWAPStrategy(target_quantity=1000.0, time_horizon=3600.0))
    manager.add_strategy("POV", POVStrategy(target_quantity=1000.0, participation_rate=0.1))
    
    # Momentum strategies
    manager.add_strategy("Momentum", MomentumStrategy())
    
    # Mean reversion strategies
    manager.add_strategy("Mean Reversion", MeanReversionStrategy())
    
    # Technical analysis strategies
    manager.add_strategy("Technical Analysis", TechnicalAnalysisStrategy())
    
    # Benchmark strategies
    manager.add_strategy("Random Walk", RandomWalkStrategy())
    manager.add_strategy("Buy and Hold", BuyAndHoldStrategy())
    
    return manager
