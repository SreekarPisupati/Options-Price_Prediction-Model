"""
Evaluation and Backtesting Framework

This module provides comprehensive evaluation and backtesting capabilities including:
- Model performance comparison against Black-Scholes baseline
- Backtesting with simulated trading
- Risk-adjusted performance metrics
- Statistical significance testing
- Performance visualization
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
import json

from data_collector import OptionsDataCollector
from feature_engineering import OptionsFeatureEngineering
from ml_models import OptionsMLModels
from black_scholes import BlackScholesModel

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison
    """
    
    def __init__(self):
        """Initialize model evaluator"""
        self.evaluation_results = {}
        self.backtesting_results = {}
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger
    
    def performance_vs_baseline(self, ml_predictions: np.ndarray, 
                              bs_predictions: np.ndarray,
                              actual_prices: np.ndarray,
                              model_name: str = "ML Model") -> Dict:
        """
        Compare ML model performance against Black-Scholes baseline
        
        Args:
            ml_predictions: ML model predictions
            bs_predictions: Black-Scholes predictions
            actual_prices: Actual option prices
            model_name: Name of ML model
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Evaluate both models
            ml_metrics = self.evaluate_model_performance(actual_prices, ml_predictions, model_name)
            bs_metrics = self.evaluate_model_performance(actual_prices, bs_predictions, "Black-Scholes")
            
            # Calculate improvement metrics
            mse_improvement = (bs_metrics['mse'] - ml_metrics['mse']) / bs_metrics['mse'] if bs_metrics['mse'] != 0 else 0.0
            mae_improvement = (bs_metrics['mae'] - ml_metrics['mae']) / bs_metrics['mae'] if bs_metrics['mae'] != 0 else 0.0
            r2_improvement = ml_metrics['r2'] - bs_metrics['r2']
            
            # Statistical significance test
            ml_errors = ml_predictions - actual_prices
            bs_errors = bs_predictions - actual_prices
            
            significance_test = self.statistical_significance_test(
                ml_errors, bs_errors, model_name, "Black-Scholes"
            )
            
            comparison_results = {
                'ml_model_metrics': ml_metrics,
                'baseline_metrics': bs_metrics,
                'mse_improvement_pct': mse_improvement * 100,
                'mae_improvement_pct': mae_improvement * 100,
                'r2_improvement': r2_improvement,
                'directional_accuracy_diff': (ml_metrics['directional_accuracy'] - 
                                            bs_metrics['directional_accuracy']),
                'statistical_test': significance_test,
                'ml_model_better': ml_metrics['mse'] < bs_metrics['mse']
            }
            
            # Log results
            if mse_improvement > 0:
                self.logger.info(f"{model_name} outperforms Black-Scholes by {mse_improvement:.1%} (MSE)")
            else:
                self.logger.info(f"Black-Scholes outperforms {model_name} by {abs(mse_improvement):.1%} (MSE)")
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing with baseline: {str(e)}")
            return {}
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "Model") -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Basic regression metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # R-squared
            r2 = r2_score(y_true, y_pred)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Directional accuracy
            actual_changes = np.diff(y_true)
            pred_changes = np.diff(y_pred)
            directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
            
            # Bias metrics
            mean_error = np.mean(y_pred - y_true)
            median_error = np.median(y_pred - y_true)
            
            # Quantile-based metrics
            errors = y_pred - y_true
            q25_error = np.percentile(errors, 25)
            q75_error = np.percentile(errors, 75)
            
            # Statistical significance
            _, p_value = stats.ttest_1samp(errors, 0)
            
            # Correlation
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'directional_accuracy': directional_accuracy,
                'mean_error': mean_error,
                'median_error': median_error,
                'q25_error': q25_error,
                'q75_error': q75_error,
                'correlation': correlation,
                'bias_p_value': p_value,
                'sample_size': len(y_true)
            }
            
            self.evaluation_results[model_name] = metrics
            self.logger.info(f"{model_name} evaluation completed - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {str(e)}")
            return {}
    
    def compare_models(self, predictions_dict: Dict[str, np.ndarray], 
                      y_true: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models' performance
        
        Args:
            predictions_dict: Dictionary with model names as keys and predictions as values
            y_true: Actual values
            
        Returns:
            DataFrame with comparison results
        """
        try:
            comparison_results = []
            
            for model_name, y_pred in predictions_dict.items():
                metrics = self.evaluate_model_performance(y_true, y_pred, model_name)
                metrics['model'] = model_name
                comparison_results.append(metrics)
            
            comparison_df = pd.DataFrame(comparison_results)
            
            # Rank models by key metrics
            comparison_df['mse_rank'] = comparison_df['mse'].rank()
            comparison_df['r2_rank'] = comparison_df['r2'].rank(ascending=False)
            comparison_df['mae_rank'] = comparison_df['mae'].rank()
            
            # Overall rank (average of key metric ranks)
            comparison_df['overall_rank'] = (comparison_df['mse_rank'] + 
                                           comparison_df['r2_rank'] + 
                                           comparison_df['mae_rank']) / 3
            
            comparison_df = comparison_df.sort_values('overall_rank')
            
            self.logger.info(f"Model comparison completed for {len(predictions_dict)} models")
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return pd.DataFrame()
    
    def statistical_significance_test(self, model1_errors: np.ndarray, 
                                    model2_errors: np.ndarray,
                                    model1_name: str = "Model 1",
                                    model2_name: str = "Model 2") -> Dict:
        """
        Test statistical significance of difference between two models
        
        Args:
            model1_errors: Prediction errors for model 1
            model2_errors: Prediction errors for model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
            
        Returns:
            Dictionary with test results
        """
        try:
            # Paired t-test
            stat, p_value = stats.ttest_rel(model1_errors, model2_errors)
            
            # Wilcoxon signed-rank test (non-parametric)
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(model1_errors, model2_errors)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(model1_errors) + np.var(model2_errors)) / 2)
            cohens_d = (np.mean(model1_errors) - np.mean(model2_errors)) / pooled_std
            
            results = {
                'model1': model1_name,
                'model2': model2_name,
                'paired_t_stat': stat,
                'paired_t_pvalue': p_value,
                'wilcoxon_stat': wilcoxon_stat,
                'wilcoxon_pvalue': wilcoxon_p,
                'cohens_d': cohens_d,
                'significant_at_05': p_value < 0.05,
                'significant_at_01': p_value < 0.01,
                'better_model': model1_name if np.mean(np.abs(model1_errors)) < np.mean(np.abs(model2_errors)) else model2_name
            }
            
            self.logger.info(f"Significance test: {model1_name} vs {model2_name} - p-value: {p_value:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in significance test: {str(e)}")
            return {}

class OptionsBacktester:
    """
    Comprehensive backtesting framework for options trading strategies
    """
    
    def __init__(self, initial_capital: float = 100000, commission: float = 1.0):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.portfolio_values = []
        self.positions = []
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger
    
    def simulate_trading_strategy(self, signals: List, option_data: Dict,
                                trading_rules: Dict = None) -> Dict:
        """
        Simulate trading strategy based on prediction signals
        
        Args:
            signals: List of prediction signals
            option_data: Historical option data
            trading_rules: Dictionary with trading rules and parameters
            
        Returns:
            Dictionary with backtest results
        """
        try:
            if trading_rules is None:
                trading_rules = {
                    'min_confidence': 0.6,
                    'max_position_size': 0.05,  # 5% of capital per trade
                    'stop_loss': -0.5,          # 50% stop loss
                    'take_profit': 1.0,         # 100% take profit
                    'max_time_to_expiry': 60    # Max 60 days to expiration
                }
            
            # Initialize portfolio
            current_capital = self.initial_capital
            positions = []
            trades = []
            daily_pnl = []
            
            # Sort signals by timestamp
            sorted_signals = sorted(signals, key=lambda x: x.timestamp)
            
            for signal in sorted_signals:
                try:
                    # Check if signal meets trading criteria
                    if not self._meets_trading_criteria(signal, trading_rules):
                        continue
                    
                    # Calculate position size
                    position_size = self._calculate_position_size(
                        signal, current_capital, trading_rules
                    )
                    
                    if position_size == 0:
                        continue
                    
                    # Execute trade
                    trade_result = self._execute_trade(signal, position_size, option_data)
                    
                    if trade_result:
                        trades.append(trade_result)
                        current_capital += trade_result['pnl']
                        
                        # Update positions
                        if trade_result['action'] == 'open':
                            positions.append(trade_result)
                        elif trade_result['action'] == 'close':
                            # Remove closed position
                            positions = [p for p in positions 
                                       if p['signal_id'] != trade_result['signal_id']]
                
                except Exception as signal_error:
                    self.logger.warning(f"Error processing signal: {str(signal_error)}")
                    continue
            
            # Calculate performance metrics
            total_return = (current_capital - self.initial_capital) / self.initial_capital
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_trade_pnl = np.mean([t['pnl'] for t in trades]) if trades else 0
            max_drawdown = self._calculate_max_drawdown([t['pnl'] for t in trades])
            
            sharpe_ratio = self._calculate_sharpe_ratio([t['pnl'] for t in trades])
            
            backtest_results = {
                'initial_capital': self.initial_capital,
                'final_capital': current_capital,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'avg_trade_pnl': avg_trade_pnl,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'trades': trades,
                'final_positions': positions
            }
            
            self.logger.info(f"Backtesting completed - Total Return: {total_return:.2%}, "
                           f"Win Rate: {win_rate:.2%}, Sharpe: {sharpe_ratio:.2f}")
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error in strategy simulation: {str(e)}")
            return {}
    
    def _meets_trading_criteria(self, signal, trading_rules: Dict) -> bool:
        """Check if signal meets trading criteria"""
        try:
            # Confidence threshold
            if signal.confidence < trading_rules.get('min_confidence', 0.6):
                return False
            
            # Signal strength
            if signal.signal_strength == 'weak':
                return False
            
            # Time to expiration
            # (Would need actual time calculation here)
            
            # Only trade BUY signals in backtest for simplicity
            if signal.recommendation not in ['BUY', 'SELL']:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_position_size(self, signal, capital: float, trading_rules: Dict) -> int:
        """Calculate position size based on capital and rules"""
        try:
            max_position_value = capital * trading_rules.get('max_position_size', 0.05)
            option_price = signal.current_price
            contract_value = option_price * 100  # Standard option contract
            
            position_size = int(max_position_value / contract_value)
            return max(position_size, 0)
            
        except Exception:
            return 0
    
    def _execute_trade(self, signal, position_size: int, option_data: Dict) -> Optional[Dict]:
        """Execute simulated trade"""
        try:
            if position_size <= 0:
                return None
            
            # Simulate trade execution
            entry_price = signal.current_price
            predicted_price = signal.predicted_price
            
            # Simplified P&L calculation
            if signal.recommendation == 'BUY':
                # Assume we hold until expiration or profit target
                expected_exit_price = predicted_price
                pnl = position_size * (expected_exit_price - entry_price) * 100
            else:
                # Sell position
                expected_exit_price = entry_price * 0.8  # Assume 20% decline
                pnl = position_size * (entry_price - expected_exit_price) * 100
            
            # Subtract commission
            pnl -= self.commission * 2  # Entry and exit commission
            
            trade_result = {
                'signal_id': f"{signal.symbol}_{signal.strike}_{signal.expiration}_{signal.option_type}",
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'option_type': signal.option_type,
                'strike': signal.strike,
                'expiration': signal.expiration,
                'action': 'open',
                'position_size': position_size,
                'entry_price': entry_price,
                'predicted_price': predicted_price,
                'exit_price': expected_exit_price,
                'pnl': pnl,
                'confidence': signal.confidence,
                'recommendation': signal.recommendation
            }
            
            return trade_result
            
        except Exception as e:
            self.logger.warning(f"Error executing trade: {str(e)}")
            return None
    
    def _calculate_max_drawdown(self, pnl_series: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if not pnl_series:
                return 0.0
            
            cumulative_pnl = np.cumsum(pnl_series)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = cumulative_pnl - running_max
            max_drawdown = np.min(drawdown)
            
            return max_drawdown
            
        except Exception:
            return 0.0
    
    def _calculate_sharpe_ratio(self, pnl_series: List[float], 
                               risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not pnl_series or len(pnl_series) < 2:
                return 0.0
            
            returns = np.array(pnl_series)
            if np.std(returns) == 0:
                return 0.0
            
            excess_return = np.mean(returns) - risk_free_rate / 252  # Daily risk-free rate
            sharpe = excess_return / np.std(returns) * np.sqrt(252)  # Annualized
            
            return sharpe
            
        except Exception:
            return 0.0
    
    def performance_vs_baseline(self, ml_predictions: np.ndarray, 
                              bs_predictions: np.ndarray,
                              actual_prices: np.ndarray,
                              model_name: str = "ML Model") -> Dict:
        """
        Compare ML model performance against Black-Scholes baseline
        
        Args:
            ml_predictions: ML model predictions
            bs_predictions: Black-Scholes predictions
            actual_prices: Actual option prices
            model_name: Name of ML model
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Evaluate both models
            ml_metrics = self.evaluate_model_performance(actual_prices, ml_predictions, model_name)
            bs_metrics = self.evaluate_model_performance(actual_prices, bs_predictions, "Black-Scholes")
            
            # Calculate improvement metrics
            mse_improvement = (bs_metrics['mse'] - ml_metrics['mse']) / bs_metrics['mse']
            mae_improvement = (bs_metrics['mae'] - ml_metrics['mae']) / bs_metrics['mae']
            r2_improvement = ml_metrics['r2'] - bs_metrics['r2']
            
            # Statistical significance test
            ml_errors = ml_predictions - actual_prices
            bs_errors = bs_predictions - actual_prices
            
            significance_test = self._statistical_significance_test(
                ml_errors, bs_errors, model_name, "Black-Scholes"
            )
            
            comparison_results = {
                'ml_model_metrics': ml_metrics,
                'baseline_metrics': bs_metrics,
                'mse_improvement_pct': mse_improvement * 100,
                'mae_improvement_pct': mae_improvement * 100,
                'r2_improvement': r2_improvement,
                'directional_accuracy_diff': (ml_metrics['directional_accuracy'] - 
                                            bs_metrics['directional_accuracy']),
                'statistical_test': significance_test,
                'ml_model_better': ml_metrics['mse'] < bs_metrics['mse']
            }
            
            # Log results
            if mse_improvement > 0:
                self.logger.info(f"{model_name} outperforms Black-Scholes by {mse_improvement:.1%} (MSE)")
            else:
                self.logger.info(f"Black-Scholes outperforms {model_name} by {abs(mse_improvement):.1%} (MSE)")
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing with baseline: {str(e)}")
            return {}
    
    def _statistical_significance_test(self, errors1: np.ndarray, errors2: np.ndarray,
                                     model1_name: str, model2_name: str) -> Dict:
        """Perform statistical significance test between two models"""
        try:
            # Paired t-test on absolute errors
            abs_errors1 = np.abs(errors1)
            abs_errors2 = np.abs(errors2)
            
            stat, p_value = stats.ttest_rel(abs_errors1, abs_errors2)
            
            # Wilcoxon signed-rank test
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(abs_errors1, abs_errors2)
            
            # Effect size
            mean_diff = np.mean(abs_errors1) - np.mean(abs_errors2)
            pooled_std = np.sqrt((np.var(abs_errors1) + np.var(abs_errors2)) / 2)
            effect_size = mean_diff / pooled_std
            
            return {
                'paired_t_stat': stat,
                'paired_t_pvalue': p_value,
                'wilcoxon_stat': wilcoxon_stat,
                'wilcoxon_pvalue': wilcoxon_p,
                'effect_size': effect_size,
                'significant_at_05': p_value < 0.05,
                'better_model': model1_name if mean_diff < 0 else model2_name
            }
            
        except Exception as e:
            self.logger.error(f"Error in statistical test: {str(e)}")
            return {}

class PerformanceAnalyzer:
    """
    Advanced performance analysis and visualization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model") -> Dict:
        """
        Analyze prediction residuals
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Model name
            
        Returns:
            Dictionary with residual analysis
        """
        try:
            residuals = y_pred - y_true
            
            # Basic statistics
            residual_stats = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'median': np.median(residuals)
            }
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
            jb_stat, jb_p = stats.jarque_bera(residuals)
            
            # Autocorrelation test
            ljung_box_stat, ljung_box_p = self._ljung_box_test(residuals)
            
            # Heteroscedasticity test
            het_test_result = self._heteroscedasticity_test(residuals, y_pred)
            
            analysis = {
                'residual_stats': residual_stats,
                'normality_tests': {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_pvalue': shapiro_p,
                    'jarque_bera_stat': jb_stat,
                    'jarque_bera_pvalue': jb_p,
                    'normal_at_05': shapiro_p > 0.05
                },
                'autocorrelation_test': {
                    'ljung_box_stat': ljung_box_stat,
                    'ljung_box_pvalue': ljung_box_p,
                    'no_autocorrelation_at_05': ljung_box_p > 0.05
                },
                'heteroscedasticity_test': het_test_result
            }
            
            self.logger.info(f"Residual analysis completed for {model_name}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in residual analysis: {str(e)}")
            return {}
    
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Tuple[float, float]:
        """Ljung-Box test for autocorrelation"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(residuals, lags=lags, return_df=True)
            return result['lb_stat'].iloc[-1], result['lb_pvalue'].iloc[-1]
        except ImportError:
            # Fallback calculation
            n = len(residuals)
            autocorrs = [np.corrcoef(residuals[:-lag], residuals[lag:])[0,1] 
                        for lag in range(1, min(lags+1, len(residuals)//4))]
            
            lb_stat = n * (n + 2) * sum([(ac**2) / (n - lag - 1) 
                                        for lag, ac in enumerate(autocorrs, 1)])
            
            # Approximate p-value
            p_value = 1 - stats.chi2.cdf(lb_stat, len(autocorrs))
            return lb_stat, p_value
        except Exception:
            return 0.0, 1.0
    
    def _heteroscedasticity_test(self, residuals: np.ndarray, predictions: np.ndarray) -> Dict:
        """Test for heteroscedasticity in residuals"""
        try:
            # Breusch-Pagan test approximation
            squared_residuals = residuals ** 2
            correlation = np.corrcoef(squared_residuals, predictions)[0, 1]
            
            # Simple test based on correlation
            n = len(residuals)
            test_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(test_stat), n - 2))
            
            return {
                'test_statistic': test_stat,
                'p_value': p_value,
                'homoscedastic_at_05': p_value > 0.05,
                'correlation_with_predictions': correlation
            }
            
        except Exception:
            return {'homoscedastic_at_05': True, 'correlation_with_predictions': 0.0}
    
    def create_performance_report(self, evaluation_results: Dict, 
                                backtest_results: Dict = None) -> str:
        """
        Create comprehensive performance report
        
        Args:
            evaluation_results: Dictionary with model evaluation results
            backtest_results: Dictionary with backtesting results
            
        Returns:
            Formatted performance report string
        """
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("OPTIONS PRICE PREDICTION MODEL - PERFORMANCE REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Model Performance Section
            report_lines.append("MODEL PERFORMANCE METRICS")
            report_lines.append("-" * 30)
            
            for model_name, metrics in evaluation_results.items():
                report_lines.append(f"\n{model_name}:")
                report_lines.append(f"  RMSE: {metrics.get('rmse', 0):.4f}")
                report_lines.append(f"  MAE:  {metrics.get('mae', 0):.4f}")
                report_lines.append(f"  R²:   {metrics.get('r2', 0):.4f}")
                report_lines.append(f"  MAPE: {metrics.get('mape', 0):.2f}%")
                report_lines.append(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}")
                report_lines.append(f"  Sample Size: {metrics.get('sample_size', 0)}")
            
            # Backtesting Section
            if backtest_results:
                report_lines.append("\n\nBACKTESTING RESULTS")
                report_lines.append("-" * 20)
                report_lines.append(f"Initial Capital: ${backtest_results.get('initial_capital', 0):,.2f}")
                report_lines.append(f"Final Capital:   ${backtest_results.get('final_capital', 0):,.2f}")
                report_lines.append(f"Total Return:    {backtest_results.get('total_return', 0):.2%}")
                report_lines.append(f"Total Trades:    {backtest_results.get('total_trades', 0)}")
                report_lines.append(f"Win Rate:        {backtest_results.get('win_rate', 0):.2%}")
                report_lines.append(f"Avg Trade P&L:   ${backtest_results.get('avg_trade_pnl', 0):,.2f}")
                report_lines.append(f"Max Drawdown:    ${backtest_results.get('max_drawdown', 0):,.2f}")
                report_lines.append(f"Sharpe Ratio:    {backtest_results.get('sharpe_ratio', 0):.2f}")
            
            # Model Ranking
            if len(evaluation_results) > 1:
                report_lines.append("\n\nMODEL RANKING (by RMSE)")
                report_lines.append("-" * 25)
                
                sorted_models = sorted(evaluation_results.items(), 
                                     key=lambda x: x[1].get('rmse', float('inf')))
                
                for i, (model_name, metrics) in enumerate(sorted_models, 1):
                    report_lines.append(f"{i}. {model_name} - RMSE: {metrics.get('rmse', 0):.4f}")
            
            report_lines.append("\n" + "=" * 60)
            
            report = "\\n".join(report_lines)
            self.logger.info("Performance report generated")
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating performance report: {str(e)}")
            return "Error generating performance report"

class OptionsBacktestingFramework:
    """
    Advanced backtesting framework with realistic trading simulation
    """
    
    def __init__(self, data_collector: OptionsDataCollector, 
                 ml_models: OptionsMLModels,
                 bs_model: BlackScholesModel):
        """
        Initialize backtesting framework
        
        Args:
            data_collector: Data collection instance
            ml_models: ML models instance
            bs_model: Black-Scholes model instance
        """
        self.data_collector = data_collector
        self.ml_models = ml_models
        self.bs_model = bs_model
        self.evaluator = ModelEvaluator()
        self.analyzer = PerformanceAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_backtest(self, symbols: List[str], 
                                 start_date: str, end_date: str,
                                 strategy_params: Dict = None) -> Dict:
        """
        Run comprehensive backtesting across multiple symbols and time periods
        
        Args:
            symbols: List of symbols to test
            start_date: Start date for backtesting
            end_date: End date for backtesting
            strategy_params: Strategy parameters
            
        Returns:
            Dictionary with comprehensive backtest results
        """
        try:
            self.logger.info(f"Starting comprehensive backtest for {len(symbols)} symbols")
            
            all_results = {
                'backtest_params': {
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'strategy_params': strategy_params
                },
                'individual_results': {},
                'aggregate_results': {},
                'model_comparison': {}
            }
            
            # Collect data for all symbols
            all_predictions = {'ml': [], 'bs': [], 'actual': []}
            
            for symbol in symbols:
                try:
                    self.logger.info(f"Processing {symbol}...")
                    
                    # Get comprehensive data
                    symbol_data = self.data_collector.collect_comprehensive_data(symbol)
                    
                    if not symbol_data or 'option_chains' not in symbol_data:
                        self.logger.warning(f"No option data for {symbol}")
                        continue
                    
                    # Process each expiration
                    symbol_results = {'ml_pred': [], 'bs_pred': [], 'actual': []}
                    
                    for exp_date, chain_data in symbol_data['option_chains'].items():
                        if 'calls' in chain_data and not chain_data['calls'].empty:
                            # Process calls
                            results = self._process_option_chain(
                                symbol_data['historical_data'],
                                chain_data['calls'],
                                symbol_data.get('sentiment_data', {})
                            )
                            
                            if results:
                                symbol_results['ml_pred'].extend(results['ml_predictions'])
                                symbol_results['bs_pred'].extend(results['bs_predictions'])
                                symbol_results['actual'].extend(results['actual_prices'])
                    
                    # Evaluate symbol-specific performance
                    if symbol_results['ml_pred']:
                        ml_pred = np.array(symbol_results['ml_pred'])
                        bs_pred = np.array(symbol_results['bs_pred'])
                        actual = np.array(symbol_results['actual'])
                        
                        symbol_comparison = self.evaluator.performance_vs_baseline(
                            ml_pred, bs_pred, actual, f"ML_Model_{symbol}"
                        )
                        
                        all_results['individual_results'][symbol] = symbol_comparison
                        
                        # Add to aggregate
                        all_predictions['ml'].extend(ml_pred)
                        all_predictions['bs'].extend(bs_pred)
                        all_predictions['actual'].extend(actual)
                
                except Exception as symbol_error:
                    self.logger.warning(f"Error processing {symbol}: {str(symbol_error)}")
                    continue
            
            # Aggregate analysis
            if all_predictions['ml']:
                ml_pred_all = np.array(all_predictions['ml'])
                bs_pred_all = np.array(all_predictions['bs'])
                actual_all = np.array(all_predictions['actual'])
                
                aggregate_comparison = self.evaluator.performance_vs_baseline(
                    ml_pred_all, bs_pred_all, actual_all, "ML_Model_Aggregate"
                )
                
                all_results['aggregate_results'] = aggregate_comparison
                
                # Calculate 12% performance boost achievement
                mse_improvement = aggregate_comparison['mse_improvement_pct']
                performance_boost_achieved = mse_improvement >= 12.0
                
                all_results['performance_boost'] = {
                    'target_boost': 12.0,
                    'achieved_boost': mse_improvement,
                    'target_achieved': performance_boost_achieved
                }
                
                self.logger.info(f"Backtesting completed. Performance boost: {mse_improvement:.1f}%")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive backtest: {str(e)}")
            return {}
    
    def _process_option_chain(self, historical_data: pd.DataFrame,
                            option_data: pd.DataFrame,
                            sentiment_data: Dict) -> Optional[Dict]:
        """Process single option chain for backtesting"""
        try:
            # Feature engineering
            feature_engineer = OptionsFeatureEngineering()
            engineered_features = feature_engineer.engineer_all_features(
                historical_data, option_data, sentiment_data
            )
            
            # Prepare features
            feature_cols = [col for col in engineered_features.columns 
                          if col not in ['symbol', 'contractSymbol', 'expiration', 
                                       'type', 'strike', 'lastTradeDate', 'mid_price']]
            
            X_features = engineered_features[feature_cols].select_dtypes(include=[np.number])
            X_features = X_features.fillna(X_features.median())
            
            if X_features.empty or 'mid_price' not in engineered_features.columns:
                return None
            
            actual_prices = engineered_features['mid_price'].values
            
            # ML predictions
            ml_predictions = None
            if self.ml_models.models:
                try:
                    ml_predictions = self.ml_models.predict(X_features, 'best')
                except Exception:
                    pass
            
            # Black-Scholes predictions
            bs_predictions_df = self.bs_model.batch_option_pricing(engineered_features)
            bs_predictions = bs_predictions_df['bs_price'].values
            
            results = {
                'actual_prices': actual_prices.tolist(),
                'bs_predictions': bs_predictions.tolist(),
                'ml_predictions': ml_predictions.tolist() if ml_predictions is not None else []
            }
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Error processing option chain: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    print("Evaluation and Backtesting Framework initialized successfully")
    
    # Create sample data for testing
    np.random.seed(42)
    y_true = np.random.randn(1000) * 10 + 50
    y_pred_ml = y_true + np.random.randn(1000) * 2  # ML predictions with some error
    y_pred_bs = y_true + np.random.randn(1000) * 3  # BS predictions with more error
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    ml_metrics = evaluator.evaluate_model_performance(y_true, y_pred_ml, "ML Model")
    bs_metrics = evaluator.evaluate_model_performance(y_true, y_pred_bs, "Black-Scholes")
    
    print(f"ML Model R²: {ml_metrics['r2']:.4f}")
    print(f"Black-Scholes R²: {bs_metrics['r2']:.4f}")
    
    # Compare models
    comparison = evaluator.performance_vs_baseline(y_pred_ml, y_pred_bs, y_true)
    print(f"MSE improvement: {comparison['mse_improvement_pct']:.1f}%")
    
    # Performance analysis
    analyzer = PerformanceAnalyzer()
    residual_analysis = analyzer.analyze_residuals(y_true, y_pred_ml, "ML Model")
    print(f"Residual normality (Shapiro p-value): {residual_analysis['normality_tests']['shapiro_pvalue']:.4f}")
    
    # Generate report
    report = analyzer.create_performance_report({
        'ML Model': ml_metrics,
        'Black-Scholes': bs_metrics
    })
    print("\\nPerformance Report Generated Successfully")
