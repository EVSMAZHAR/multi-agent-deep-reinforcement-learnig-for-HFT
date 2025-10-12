"""
Comprehensive Evaluation Framework for Multi-Agent High-Frequency Trading
=========================================================================

This module provides a comprehensive evaluation framework for comparing
multi-agent reinforcement learning algorithms with baseline strategies:

1. Performance Metrics (Sharpe, Sortino, Calmar, etc.)
2. Risk Metrics (VaR, CVaR, Maximum Drawdown, etc.)
3. Statistical Significance Testing
4. Bootstrap Confidence Intervals
5. Multi-Agent Specific Metrics
6. Market Impact Analysis
7. Execution Quality Metrics
8. Robustness Testing

Based on the thesis: "Multi-agent deep reinforcement learning for high frequency trading"
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework"""
    # Statistical testing
    confidence_level: float = 0.95
    bootstrap_samples: int = 10000
    significance_level: float = 0.05
    
    # Risk metrics
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    lookback_period: int = 252  # Trading days
    
    # Performance metrics
    risk_free_rate: float = 0.02
    benchmark_return: float = 0.08
    
    # Multi-agent specific
    coordination_penalty: float = 0.1
    competition_reward: float = 0.05
    
    # Market impact
    impact_horizon: int = 10  # minutes
    permanent_impact_weight: float = 0.7
    temporary_impact_weight: float = 0.3


class PerformanceMetrics:
    """Comprehensive performance metrics calculator"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def calculate_all_metrics(self, returns: np.ndarray, benchmark_returns: np.ndarray = None) -> Dict[str, float]:
        """Calculate all performance metrics"""
        metrics = {}
        
        # Basic statistics
        metrics.update(self._basic_statistics(returns))
        
        # Risk-adjusted returns
        metrics.update(self._risk_adjusted_returns(returns))
        
        # Risk metrics
        metrics.update(self._risk_metrics(returns))
        
        # Drawdown metrics
        metrics.update(self._drawdown_metrics(returns))
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics.update(self._benchmark_comparison(returns, benchmark_returns))
        
        return metrics
    
    def _basic_statistics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical metrics"""
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'median_return': np.median(returns),
            'total_return': np.sum(returns),
            'positive_returns': np.sum(returns > 0),
            'negative_returns': np.sum(returns < 0),
            'win_rate': np.mean(returns > 0)
        }
    
    def _risk_adjusted_returns(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe ratio
        sharpe = (mean_return - self.config.risk_free_rate / self.config.lookback_period) / (std_return + 1e-8)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
        sortino = (mean_return - self.config.risk_free_rate / self.config.lookback_period) / (downside_std + 1e-8)
        
        # Calmar ratio
        max_dd = self._calculate_max_drawdown(returns)
        calmar = mean_return / (abs(max_dd) + 1e-8)
        
        # Information ratio (if benchmark available)
        info_ratio = None  # Will be calculated in benchmark comparison
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'information_ratio': info_ratio
        }
    
    def _risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics"""
        # Value at Risk (VaR)
        var = np.percentile(returns, (1 - self.config.var_confidence) * 100)
        
        # Conditional Value at Risk (CVaR)
        cvar_threshold = np.percentile(returns, (1 - self.config.cvar_confidence) * 100)
        cvar = np.mean(returns[returns <= cvar_threshold]) if np.any(returns <= cvar_threshold) else var
        
        # Expected Shortfall
        expected_shortfall = cvar
        
        # Tail risk metrics
        tail_ratio = self._calculate_tail_ratio(returns)
        
        return {
            'var_95': var,
            'cvar_95': cvar,
            'expected_shortfall': expected_shortfall,
            'tail_ratio': tail_ratio
        }
    
    def _drawdown_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown-related metrics"""
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        
        max_dd = np.min(drawdowns)
        max_dd_duration = self._calculate_max_drawdown_duration(drawdowns)
        avg_drawdown = np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0
        
        # Recovery time
        recovery_time = self._calculate_recovery_time(drawdowns)
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_duration': max_dd_duration,
            'avg_drawdown': avg_drawdown,
            'recovery_time': recovery_time
        }
    
    def _benchmark_comparison(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> Dict[str, float]:
        """Compare with benchmark"""
        # Information ratio
        excess_returns = returns - benchmark_returns
        info_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
        
        # Beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / (benchmark_variance + 1e-8)
        
        # Alpha
        alpha = np.mean(returns) - self.config.risk_free_rate / self.config.lookback_period - beta * (
            np.mean(benchmark_returns) - self.config.risk_free_rate / self.config.lookback_period
        )
        
        # Tracking error
        tracking_error = np.std(excess_returns)
        
        # Correlation
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        
        return {
            'information_ratio': info_ratio,
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'correlation': correlation
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        return np.min(drawdowns)
    
    def _calculate_max_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """Calculate maximum drawdown duration"""
        in_drawdown = drawdowns < 0
        if not np.any(in_drawdown):
            return 0
        
        # Find consecutive drawdown periods
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_recovery_time(self, drawdowns: np.ndarray) -> float:
        """Calculate average recovery time from drawdowns"""
        in_drawdown = drawdowns < 0
        if not np.any(in_drawdown):
            return 0
        
        recovery_times = []
        current_recovery = 0
        
        for i, is_dd in enumerate(in_drawdown):
            if not is_dd and current_recovery > 0:
                recovery_times.append(current_recovery)
                current_recovery = 0
            elif is_dd:
                current_recovery += 1
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return p95 / (abs(p5) + 1e-8)


class MultiAgentMetrics:
    """Multi-agent specific metrics"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def calculate_coordination_metrics(
        self,
        agent_returns: Dict[str, np.ndarray],
        joint_returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate multi-agent coordination metrics"""
        metrics = {}
        
        # Individual agent performance
        individual_sharpes = {}
        for agent_id, returns in agent_returns.items():
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
            individual_sharpes[agent_id] = sharpe
        
        metrics['individual_sharpes'] = individual_sharpes
        metrics['avg_individual_sharpe'] = np.mean(list(individual_sharpes.values()))
        
        # Joint performance
        joint_sharpe = np.mean(joint_returns) / (np.std(joint_returns) + 1e-8)
        metrics['joint_sharpe'] = joint_sharpe
        
        # Coordination efficiency
        coordination_efficiency = joint_sharpe / (metrics['avg_individual_sharpe'] + 1e-8)
        metrics['coordination_efficiency'] = coordination_efficiency
        
        # Diversity index
        diversity_index = self._calculate_diversity_index(agent_returns)
        metrics['diversity_index'] = diversity_index
        
        # Competition index
        competition_index = self._calculate_competition_index(agent_returns)
        metrics['competition_index'] = competition_index
        
        return metrics
    
    def calculate_market_impact_metrics(
        self,
        executions: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate market impact metrics"""
        if not executions:
            return {}
        
        # Separate permanent and temporary impact
        permanent_impacts = []
        temporary_impacts = []
        
        for execution in executions:
            if 'permanent_impact' in execution:
                permanent_impacts.append(execution['permanent_impact'])
            if 'temporary_impact' in execution:
                temporary_impacts.append(execution['temporary_impact'])
        
        # Calculate impact metrics
        metrics = {
            'avg_permanent_impact': np.mean(permanent_impacts) if permanent_impacts else 0,
            'avg_temporary_impact': np.mean(temporary_impacts) if temporary_impacts else 0,
            'total_impact': np.mean(permanent_impacts) + np.mean(temporary_impacts) if permanent_impacts and temporary_impacts else 0
        }
        
        # Impact decay analysis
        if len(executions) > 1:
            impact_decay = self._calculate_impact_decay(executions, market_data)
            metrics['impact_decay_rate'] = impact_decay
        
        return metrics
    
    def _calculate_diversity_index(self, agent_returns: Dict[str, np.ndarray]) -> float:
        """Calculate diversity index based on return correlations"""
        if len(agent_returns) < 2:
            return 0
        
        returns_matrix = np.array([returns for returns in agent_returns.values()])
        correlation_matrix = np.corrcoef(returns_matrix)
        
        # Remove diagonal elements
        mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        correlations = correlation_matrix[mask]
        
        # Diversity is inverse of average correlation
        avg_correlation = np.mean(correlations)
        diversity_index = 1 - avg_correlation
        
        return diversity_index
    
    def _calculate_competition_index(self, agent_returns: Dict[str, np.ndarray]) -> float:
        """Calculate competition index based on performance variance"""
        individual_performances = [np.mean(returns) for returns in agent_returns.values()]
        performance_variance = np.var(individual_performances)
        
        # Normalize by average performance
        avg_performance = np.mean(individual_performances)
        competition_index = performance_variance / (avg_performance**2 + 1e-8)
        
        return competition_index
    
    def _calculate_impact_decay(
        self,
        executions: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> float:
        """Calculate market impact decay rate"""
        # This is a simplified implementation
        # In practice, you would analyze price recovery after each execution
        return 0.1  # Placeholder


class StatisticalTesting:
    """Statistical significance testing framework"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable,
        n_bootstrap: int = None
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if n_bootstrap is None:
            n_bootstrap = self.config.bootstrap_samples
        
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        # Calculate confidence interval
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    def t_test_comparison(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """Perform t-test comparison between two return series"""
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(returns_a, returns_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(returns_a) + np.var(returns_b)) / 2)
        cohens_d = (np.mean(returns_a) - np.mean(returns_b)) / (pooled_std + 1e-8)
        
        # Confidence interval for difference
        diff = returns_a - returns_b
        ci_lower, ci_upper = self.bootstrap_confidence_interval(diff, np.mean)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_difference': np.mean(diff),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < self.config.significance_level
        }
    
    def mann_whitney_test(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray
    ) -> Dict[str, float]:
        """Perform Mann-Whitney U test (non-parametric)"""
        u_stat, p_value = stats.mannwhitneyu(returns_a, returns_b, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(returns_a), len(returns_b)
        r = 1 - (2 * u_stat) / (n1 * n2)
        
        return {
            'u_statistic': u_stat,
            'p_value': p_value,
            'effect_size': r,
            'significant': p_value < self.config.significance_level
        }
    
    def normality_tests(self, returns: np.ndarray) -> Dict[str, Any]:
        """Perform normality tests"""
        results = {}
        
        # Shapiro-Wilk test
        if len(returns) <= 5000:  # Shapiro-Wilk is only valid for small samples
            shapiro_stat, shapiro_p = shapiro(returns)
            results['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'normal': shapiro_p > self.config.significance_level
            }
        
        # Jarque-Bera test
        jb_stat, jb_p = jarque_bera(returns)
        results['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'normal': jb_p > self.config.significance_level
        }
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = kstest(returns, 'norm', args=(np.mean(returns), np.std(returns)))
        results['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'normal': ks_p > self.config.significance_level
        }
        
        return results


class RobustnessTesting:
    """Robustness testing framework"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    def parameter_sensitivity_analysis(
        self,
        returns: np.ndarray,
        parameter_ranges: Dict[str, List[float]],
        metric_func: callable
    ) -> Dict[str, Dict[str, float]]:
        """Perform parameter sensitivity analysis"""
        results = {}
        
        for param_name, param_values in parameter_ranges.items():
            param_results = {}
            
            for param_value in param_values:
                # This is a placeholder - in practice, you would re-run the algorithm
                # with different parameter values
                metric_value = metric_func(returns)  # Simplified
                param_results[param_value] = metric_value
            
            results[param_name] = param_results
        
        return results
    
    def out_of_sample_testing(
        self,
        train_returns: np.ndarray,
        test_returns: np.ndarray,
        metric_func: callable
    ) -> Dict[str, float]:
        """Perform out-of-sample testing"""
        train_metric = metric_func(train_returns)
        test_metric = metric_func(test_returns)
        
        # Calculate degradation
        degradation = (test_metric - train_metric) / (abs(train_metric) + 1e-8)
        
        return {
            'train_metric': train_metric,
            'test_metric': test_metric,
            'degradation': degradation,
            'robust': abs(degradation) < 0.2  # 20% degradation threshold
        }
    
    def cross_validation_analysis(
        self,
        returns: np.ndarray,
        n_folds: int = 5,
        metric_func: callable = None
    ) -> Dict[str, float]:
        """Perform cross-validation analysis"""
        if metric_func is None:
            metric_func = lambda x: np.mean(x) / (np.std(x) + 1e-8)  # Sharpe ratio
        
        fold_size = len(returns) // n_folds
        fold_metrics = []
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else len(returns)
            
            fold_returns = returns[start_idx:end_idx]
            fold_metric = metric_func(fold_returns)
            fold_metrics.append(fold_metric)
        
        return {
            'mean_metric': np.mean(fold_metrics),
            'std_metric': np.std(fold_metrics),
            'cv_score': np.std(fold_metrics) / (np.mean(fold_metrics) + 1e-8),
            'fold_metrics': fold_metrics
        }


class ComprehensiveEvaluator:
    """Main evaluation class that orchestrates all evaluation components"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        
        # Initialize components
        self.performance_metrics = PerformanceMetrics(config)
        self.multi_agent_metrics = MultiAgentMetrics(config)
        self.statistical_testing = StatisticalTesting(config)
        self.robustness_testing = RobustnessTesting(config)
        
        # Results storage
        self.evaluation_results = {}
    
    def evaluate_strategy(
        self,
        strategy_name: str,
        returns: np.ndarray,
        benchmark_returns: np.ndarray = None,
        agent_returns: Dict[str, np.ndarray] = None,
        executions: List[Dict[str, Any]] = None,
        market_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of a single strategy"""
        
        results = {
            'strategy_name': strategy_name,
            'evaluation_timestamp': pd.Timestamp.now()
        }
        
        # Performance metrics
        performance_metrics = self.performance_metrics.calculate_all_metrics(returns, benchmark_returns)
        results['performance_metrics'] = performance_metrics
        
        # Multi-agent specific metrics
        if agent_returns is not None:
            multi_agent_metrics = self.multi_agent_metrics.calculate_coordination_metrics(
                agent_returns, returns
            )
            results['multi_agent_metrics'] = multi_agent_metrics
        
        # Market impact metrics
        if executions is not None and market_data is not None:
            impact_metrics = self.multi_agent_metrics.calculate_market_impact_metrics(
                executions, market_data
            )
            results['market_impact_metrics'] = impact_metrics
        
        # Statistical tests
        if benchmark_returns is not None:
            statistical_tests = self.statistical_testing.t_test_comparison(returns, benchmark_returns)
            results['statistical_tests'] = statistical_tests
        
        # Normality tests
        normality_tests = self.statistical_testing.normality_tests(returns)
        results['normality_tests'] = normality_tests
        
        # Robustness tests
        robustness_tests = self.robustness_testing.cross_validation_analysis(returns)
        results['robustness_tests'] = robustness_tests
        
        # Store results
        self.evaluation_results[strategy_name] = results
        
        return results
    
    def compare_strategies(
        self,
        strategy_results: Dict[str, Dict[str, Any]],
        reference_strategy: str = None
    ) -> pd.DataFrame:
        """Compare multiple strategies"""
        
        comparison_data = []
        
        for strategy_name, results in strategy_results.items():
            performance_metrics = results['performance_metrics']
            
            row = {
                'Strategy': strategy_name,
                'Sharpe Ratio': performance_metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': performance_metrics.get('sortino_ratio', 0),
                'Calmar Ratio': performance_metrics.get('calmar_ratio', 0),
                'Max Drawdown': performance_metrics.get('max_drawdown', 0),
                'VaR 95%': performance_metrics.get('var_95', 0),
                'CVaR 95%': performance_metrics.get('cvar_95', 0),
                'Win Rate': performance_metrics.get('win_rate', 0),
                'Total Return': performance_metrics.get('total_return', 0)
            }
            
            # Add multi-agent metrics if available
            if 'multi_agent_metrics' in results:
                ma_metrics = results['multi_agent_metrics']
                row['Coordination Efficiency'] = ma_metrics.get('coordination_efficiency', 0)
                row['Diversity Index'] = ma_metrics.get('diversity_index', 0)
                row['Competition Index'] = ma_metrics.get('competition_index', 0)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank strategies
        comparison_df['Overall Rank'] = comparison_df['Sharpe Ratio'].rank(ascending=False)
        
        return comparison_df
    
    def generate_report(
        self,
        strategy_results: Dict[str, Dict[str, Any]],
        output_file: str = None
    ) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE EVALUATION REPORT")
        report.append("Multi-Agent Deep Reinforcement Learning for High-Frequency Trading")
        report.append("=" * 80)
        report.append("")
        
        # Executive summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        
        # Find best performing strategy
        best_strategy = max(
            strategy_results.items(),
            key=lambda x: x[1]['performance_metrics'].get('sharpe_ratio', 0)
        )
        
        report.append(f"Best Performing Strategy: {best_strategy[0]}")
        report.append(f"Sharpe Ratio: {best_strategy[1]['performance_metrics'].get('sharpe_ratio', 0):.4f}")
        report.append(f"Max Drawdown: {best_strategy[1]['performance_metrics'].get('max_drawdown', 0):.4f}")
        report.append("")
        
        # Detailed results for each strategy
        for strategy_name, results in strategy_results.items():
            report.append(f"STRATEGY: {strategy_name}")
            report.append("-" * 40)
            
            # Performance metrics
            perf_metrics = results['performance_metrics']
            report.append("Performance Metrics:")
            report.append(f"  Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.4f}")
            report.append(f"  Sortino Ratio: {perf_metrics.get('sortino_ratio', 0):.4f}")
            report.append(f"  Calmar Ratio: {perf_metrics.get('calmar_ratio', 0):.4f}")
            report.append(f"  Max Drawdown: {perf_metrics.get('max_drawdown', 0):.4f}")
            report.append(f"  VaR 95%: {perf_metrics.get('var_95', 0):.4f}")
            report.append(f"  Win Rate: {perf_metrics.get('win_rate', 0):.4f}")
            report.append("")
            
            # Multi-agent metrics
            if 'multi_agent_metrics' in results:
                ma_metrics = results['multi_agent_metrics']
                report.append("Multi-Agent Metrics:")
                report.append(f"  Coordination Efficiency: {ma_metrics.get('coordination_efficiency', 0):.4f}")
                report.append(f"  Diversity Index: {ma_metrics.get('diversity_index', 0):.4f}")
                report.append(f"  Competition Index: {ma_metrics.get('competition_index', 0):.4f}")
                report.append("")
            
            # Statistical significance
            if 'statistical_tests' in results:
                stat_tests = results['statistical_tests']
                report.append("Statistical Tests:")
                report.append(f"  T-test p-value: {stat_tests.get('p_value', 0):.4f}")
                report.append(f"  Significant: {stat_tests.get('significant', False)}")
                report.append("")
        
        # Strategy comparison
        report.append("STRATEGY COMPARISON")
        report.append("-" * 20)
        comparison_df = self.compare_strategies(strategy_results)
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 15)
        report.append("1. Consider the best performing strategy for production deployment")
        report.append("2. Monitor risk metrics closely during live trading")
        report.append("3. Implement proper risk management controls")
        report.append("4. Regular re-evaluation and model updates recommended")
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_performance_comparison(
        self,
        strategy_results: Dict[str, Dict[str, Any]],
        save_path: str = None
    ):
        """Create performance comparison plots"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Comparison Across Strategies', fontsize=16)
        
        # Extract data for plotting
        strategy_names = list(strategy_results.keys())
        sharpe_ratios = [results['performance_metrics'].get('sharpe_ratio', 0) for results in strategy_results.values()]
        max_drawdowns = [results['performance_metrics'].get('max_drawdown', 0) for results in strategy_results.values()]
        win_rates = [results['performance_metrics'].get('win_rate', 0) for results in strategy_results.values()]
        total_returns = [results['performance_metrics'].get('total_return', 0) for results in strategy_results.values()]
        
        # Sharpe Ratio comparison
        axes[0, 0].bar(strategy_names, sharpe_ratios, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Sharpe Ratio Comparison')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Max Drawdown comparison
        axes[0, 1].bar(strategy_names, max_drawdowns, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Maximum Drawdown Comparison')
        axes[0, 1].set_ylabel('Max Drawdown')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Win Rate comparison
        axes[1, 0].bar(strategy_names, win_rates, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Win Rate Comparison')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Total Return comparison
        axes[1, 1].bar(strategy_names, total_returns, color='gold', alpha=0.7)
        axes[1, 1].set_title('Total Return Comparison')
        axes[1, 1].set_ylabel('Total Return')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation results"""
        summary = {
            'total_strategies_evaluated': len(self.evaluation_results),
            'evaluation_timestamp': pd.Timestamp.now(),
            'best_strategy': None,
            'worst_strategy': None,
            'average_sharpe': 0,
            'average_drawdown': 0
        }
        
        if self.evaluation_results:
            # Find best and worst strategies
            best_strategy = max(
                self.evaluation_results.items(),
                key=lambda x: x[1]['performance_metrics'].get('sharpe_ratio', 0)
            )
            worst_strategy = min(
                self.evaluation_results.items(),
                key=lambda x: x[1]['performance_metrics'].get('sharpe_ratio', 0)
            )
            
            summary['best_strategy'] = best_strategy[0]
            summary['worst_strategy'] = worst_strategy[0]
            
            # Calculate averages
            sharpe_ratios = [results['performance_metrics'].get('sharpe_ratio', 0) for results in self.evaluation_results.values()]
            drawdowns = [results['performance_metrics'].get('max_drawdown', 0) for results in self.evaluation_results.values()]
            
            summary['average_sharpe'] = np.mean(sharpe_ratios)
            summary['average_drawdown'] = np.mean(drawdowns)
        
        return summary
