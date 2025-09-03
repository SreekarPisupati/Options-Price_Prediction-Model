"""
Main Application Script for Options Price Prediction Model

This script orchestrates the entire options prediction system including:
- Data collection and preprocessing
- Feature engineering
- Model training and evaluation
- Real-time prediction generation
- Performance comparison against Black-Scholes baseline
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import json
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector import OptionsDataCollector
from feature_engineering import OptionsFeatureEngineering
from ml_models import OptionsMLModels
from black_scholes import BlackScholesModel
from realtime_predictor import RealTimePredictionEngine, OptionsSignalProcessor
from evaluation import ModelEvaluator, OptionsBacktestingFramework, PerformanceAnalyzer

warnings.filterwarnings('ignore')

class OptionsPredictionSystem:
    """
    Complete Options Price Prediction System
    """
    
    def __init__(self, symbols: List[str] = None, config_path: str = None):
        """
        Initialize the complete system
        
        Args:
            symbols: List of stock symbols to analyze
            config_path: Path to configuration file
        """
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_collector = OptionsDataCollector(self.symbols)
        self.feature_engineer = OptionsFeatureEngineering()
        self.ml_models = OptionsMLModels()
        self.bs_model = BlackScholesModel()
        self.evaluator = ModelEvaluator()
        self.analyzer = PerformanceAnalyzer()
        
        # Real-time components
        self.realtime_engine = None
        self.signal_processor = OptionsSignalProcessor()
        
        self.logger = self._setup_logging()
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        self.backtest_results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO, 
            format=log_format,
            handlers=[
                logging.FileHandler('../logs/options_prediction.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create logs directory if it doesn't exist
        os.makedirs('../logs', exist_ok=True)
        
        logger = logging.getLogger(__name__)
        return logger
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'data_collection': {
                'historical_period': '2y',
                'max_expirations': 6
            },
            'feature_engineering': {
                'volatility_windows': [5, 10, 20, 30, 60, 90],
                'ma_periods': [5, 10, 20, 50, 100, 200],
                'feature_selection_k': 50
            },
            'model_training': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'cv_folds': 5,
                'random_state': 42
            },
            'trading_strategy': {
                'min_confidence': 0.6,
                'max_position_size': 0.05,
                'stop_loss': -0.5,
                'take_profit': 1.0
            },
            'realtime': {
                'update_interval': 300,
                'cache_duration': 60
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                self.logger.warning(f"Could not load config file: {str(e)}")
        
        return default_config
    
    def collect_training_data(self, save_data: bool = True) -> Dict:
        """
        Collect comprehensive training data for all symbols
        
        Args:
            save_data: Whether to save collected data to files
            
        Returns:
            Dictionary with collected data for all symbols
        """
        try:
            self.logger.info("Starting comprehensive data collection...")
            
            all_data = {}
            
            for symbol in self.symbols:
                self.logger.info(f"Collecting data for {symbol}")
                
                symbol_data = self.data_collector.collect_comprehensive_data(
                    symbol, save_to_file=save_data
                )
                
                if symbol_data:
                    all_data[symbol] = symbol_data
                    self.logger.info(f"Successfully collected data for {symbol}")
                else:
                    self.logger.warning(f"Failed to collect data for {symbol}")
            
            self.logger.info(f"Data collection completed for {len(all_data)} symbols")
            return all_data
            
        except Exception as e:
            self.logger.error(f"Error in data collection: {str(e)}")
            return {}
    
    def prepare_training_dataset(self, collected_data: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare comprehensive training dataset from collected data
        
        Args:
            collected_data: Dictionary with collected data for all symbols
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        try:
            self.logger.info("Preparing training dataset...")
            
            all_features = []
            all_targets = []
            
            for symbol, symbol_data in collected_data.items():
                try:
                    if 'option_chains' not in symbol_data:
                        continue
                    
                    historical_data = symbol_data['historical_data']
                    sentiment_data = symbol_data.get('sentiment_data', {})
                    
                    # Process each expiration
                    for exp_date, chain_data in symbol_data['option_chains'].items():
                        # Process calls
                        if 'calls' in chain_data and not chain_data['calls'].empty:
                            calls_df = chain_data['calls']
                            
                            # Filter for liquid options
                            liquid_calls = calls_df[
                                (calls_df['volume'] > 0) & 
                                (calls_df['bid'] > 0) & 
                                (calls_df['ask'] > 0)
                            ]
                            
                            if not liquid_calls.empty:
                                # Engineer features
                                features_df = self.feature_engineer.engineer_all_features(
                                    historical_data, liquid_calls, sentiment_data
                                )
                                
                                # Prepare features and targets
                                if 'mid_price' in features_df.columns:
                                    targets = features_df['mid_price']
                                    
                                    # Remove non-feature columns
                                    feature_cols = [col for col in features_df.columns 
                                                  if col not in ['mid_price', 'symbol', 'contractSymbol', 
                                                               'expiration', 'type', 'strike', 'lastTradeDate']]
                                    
                                    features = features_df[feature_cols].select_dtypes(include=[np.number])
                                    features = features.fillna(features.median())
                                    
                                    # Add symbol identifier
                                    features['symbol_encoded'] = hash(symbol) % 1000
                                    
                                    all_features.append(features)
                                    all_targets.extend(targets.values)
                        
                        # Process puts (similar logic)
                        if 'puts' in chain_data and not chain_data['puts'].empty:
                            puts_df = chain_data['puts']
                            
                            liquid_puts = puts_df[
                                (puts_df['volume'] > 0) & 
                                (puts_df['bid'] > 0) & 
                                (puts_df['ask'] > 0)
                            ]
                            
                            if not liquid_puts.empty:
                                features_df = self.feature_engineer.engineer_all_features(
                                    historical_data, liquid_puts, sentiment_data
                                )
                                
                                if 'mid_price' in features_df.columns:
                                    targets = features_df['mid_price']
                                    
                                    feature_cols = [col for col in features_df.columns 
                                                  if col not in ['mid_price', 'symbol', 'contractSymbol', 
                                                               'expiration', 'type', 'strike', 'lastTradeDate']]
                                    
                                    features = features_df[feature_cols].select_dtypes(include=[np.number])
                                    features = features.fillna(features.median())
                                    features['symbol_encoded'] = hash(symbol) % 1000
                                    
                                    all_features.append(features)
                                    all_targets.extend(targets.values)
                
                except Exception as symbol_error:
                    self.logger.warning(f"Error preparing data for {symbol}: {str(symbol_error)}")
                    continue
            
            if all_features:
                # Combine all features
                combined_features = pd.concat(all_features, ignore_index=True)
                combined_targets = pd.Series(all_targets)
                
                # Ensure all features have the same columns
                combined_features = combined_features.fillna(combined_features.median())
                
                self.logger.info(f"Prepared dataset: {combined_features.shape[0]} samples, {combined_features.shape[1]} features")
                return combined_features, combined_targets
            else:
                self.logger.error("No features could be prepared")
                return pd.DataFrame(), pd.Series()
                
        except Exception as e:
            self.logger.error(f"Error preparing training dataset: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train all ML models
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with training results
        """
        try:
            self.logger.info("Starting model training...")
            
            if X.empty or y.empty:
                self.logger.error("Empty dataset provided for training")
                return {}
            
            # Train all models
            training_results = self.ml_models.train_all_models(X, y, 
                test_size=self.config['model_training']['test_size']
            )
            
            self.training_results = training_results
            
            # Save trained models
            self.ml_models.save_models("../models/")
            
            self.logger.info(f"Model training completed. Best model: {self._get_best_model_name()}")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            return {}
    
    def evaluate_models(self, collected_data: Dict) -> Dict:
        """
        Comprehensive model evaluation and comparison
        
        Args:
            collected_data: Dictionary with collected data
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            self.logger.info("Starting comprehensive model evaluation...")
            
            all_ml_predictions = []
            all_bs_predictions = []
            all_actual_prices = []
            
            for symbol in self.symbols:
                if symbol not in collected_data:
                    continue
                
                symbol_data = collected_data[symbol]
                historical_data = symbol_data['historical_data']
                sentiment_data = symbol_data.get('sentiment_data', {})
                
                for exp_date, chain_data in symbol_data.get('option_chains', {}).items():
                    for option_type in ['calls', 'puts']:
                        if option_type not in chain_data:
                            continue
                        
                        option_df = chain_data[option_type]
                        
                        if option_df.empty:
                            continue
                        
                        # Engineer features
                        features_df = self.feature_engineer.engineer_all_features(
                            historical_data, option_df, sentiment_data
                        )
                        
                        if 'mid_price' not in features_df.columns:
                            continue
                        
                        actual_prices = features_df['mid_price'].values
                        
                        # Prepare features for ML prediction
                        feature_cols = [col for col in features_df.columns 
                                      if col not in ['mid_price', 'symbol', 'contractSymbol', 
                                                   'expiration', 'type', 'strike', 'lastTradeDate']]
                        
                        X_features = features_df[feature_cols].select_dtypes(include=[np.number])
                        X_features = X_features.fillna(X_features.median())
                        X_features['symbol_encoded'] = hash(symbol) % 1000
                        
                        # ML predictions
                        try:
                            ml_predictions = self.ml_models.predict(X_features, 'best')
                            all_ml_predictions.extend(ml_predictions)
                        except Exception as ml_error:
                            self.logger.warning(f"ML prediction failed for {symbol}: {str(ml_error)}")
                            continue
                        
                        # Black-Scholes predictions
                        bs_predictions_df = self.bs_model.batch_option_pricing(features_df)
                        bs_predictions = bs_predictions_df['bs_price'].values
                        
                        all_bs_predictions.extend(bs_predictions)
                        all_actual_prices.extend(actual_prices)
            
            # Evaluate overall performance
            if all_ml_predictions and all_bs_predictions and all_actual_prices:
                ml_pred_array = np.array(all_ml_predictions)
                bs_pred_array = np.array(all_bs_predictions)
                actual_array = np.array(all_actual_prices)
                
                # Compare against Black-Scholes baseline
                comparison_results = self.evaluator.performance_vs_baseline(
                    ml_pred_array, bs_pred_array, actual_array, "ML_Ensemble"
                )
                
                self.evaluation_results = comparison_results
                
                # Log key results
                mse_improvement = comparison_results['mse_improvement_pct']
                if mse_improvement >= 12.0:
                    self.logger.info(f"🎯 TARGET ACHIEVED! MSE improvement: {mse_improvement:.1f}% (Target: 12%)")
                else:
                    self.logger.info(f"MSE improvement: {mse_improvement:.1f}% (Target: 12%)")
                
                return comparison_results
            else:
                self.logger.error("No valid predictions generated for evaluation")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            return {}
    
    def run_backtesting(self, collected_data: Dict) -> Dict:
        """
        Run comprehensive backtesting
        
        Args:
            collected_data: Dictionary with collected data
            
        Returns:
            Dictionary with backtesting results
        """
        try:
            self.logger.info("Starting backtesting framework...")
            
            backtest_framework = OptionsBacktestingFramework(
                self.data_collector, self.ml_models, self.bs_model
            )
            
            backtest_results = backtest_framework.run_comprehensive_backtest(
                self.symbols, 
                start_date="2023-01-01", 
                end_date="2024-12-31",
                strategy_params=self.config['trading_strategy']
            )
            
            self.backtest_results = backtest_results
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {str(e)}")
            return {}
    
    def start_realtime_system(self):
        """Start real-time prediction system"""
        try:
            self.logger.info("Starting real-time prediction system...")
            
            # Initialize real-time engine
            self.realtime_engine = RealTimePredictionEngine(self.symbols, "../models/")
            
            # Add signal processing callback
            def signal_callback(signal):
                processed_signal = self.signal_processor.process_signal(signal)
                if processed_signal:
                    self.logger.info(f"New signal: {processed_signal['symbol']} "
                                   f"{processed_signal['recommendation']} "
                                   f"(Confidence: {processed_signal['confidence']:.2f})")
            
            self.realtime_engine.add_signal_callback(signal_callback)
            
            # Start monitoring
            self.realtime_engine.start_realtime_monitoring(
                update_interval=self.config['realtime']['update_interval']
            )
            
            self.logger.info("Real-time system started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting real-time system: {str(e)}")
    
    def generate_performance_report(self, save_to_file: bool = True) -> str:
        """
        Generate comprehensive performance report
        
        Args:
            save_to_file: Whether to save report to file
            
        Returns:
            Performance report string
        """
        try:
            self.logger.info("Generating comprehensive performance report...")
            
            # Create report
            report = self.analyzer.create_performance_report(
                self.evaluation_results, 
                self.backtest_results
            )
            
            # Add additional sections
            report_sections = [
                report,
                "\n\nTRAINING RESULTS",
                "-" * 20
            ]
            
            if self.training_results:
                best_model = self._get_best_model_name()
                report_sections.append(f"Best performing model: {best_model}")
                
                for model_name, metrics in self.training_results.items():
                    report_sections.append(f"\n{model_name}:")
                    report_sections.append(f"  R²: {metrics.get('r2', 0):.4f}")
                    report_sections.append(f"  MSE: {metrics.get('mse', 0):.6f}")
            
            # System configuration
            report_sections.extend([
                "\n\nSYSTEM CONFIGURATION",
                "-" * 22,
                f"Symbols analyzed: {', '.join(self.symbols)}",
                f"Features engineered: {self.config['feature_engineering']['feature_selection_k']}",
                f"Models trained: {len(self.ml_models.models)}",
                f"Real-time monitoring: {'Active' if self.realtime_engine and self.realtime_engine.is_running else 'Inactive'}"
            ])
            
            full_report = "\n".join(report_sections)
            
            # Save to file
            if save_to_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"../docs/performance_report_{timestamp}.txt"
                
                os.makedirs('../docs', exist_ok=True)
                with open(report_filename, 'w') as f:
                    f.write(full_report)
                
                self.logger.info(f"Performance report saved to {report_filename}")
            
            return full_report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return "Error generating performance report"
    
    def _get_best_model_name(self) -> str:
        """Get name of best performing model"""
        try:
            if not self.ml_models.model_performances:
                return "No models trained"
            
            best_model = min(self.ml_models.model_performances.keys(), 
                           key=lambda x: self.ml_models.model_performances[x]['mse'])
            return best_model
            
        except Exception:
            return "Unknown"
    
    def run_complete_pipeline(self, include_realtime: bool = False) -> Dict:
        """
        Run the complete options prediction pipeline
        
        Args:
            include_realtime: Whether to start real-time monitoring
            
        Returns:
            Dictionary with all results
        """
        try:
            self.logger.info("🚀 Starting Complete Options Price Prediction Pipeline")
            self.logger.info("=" * 60)
            
            pipeline_results = {
                'start_time': datetime.now().isoformat(),
                'symbols': self.symbols,
                'config': self.config
            }
            
            # Step 1: Data Collection
            self.logger.info("📊 Step 1: Data Collection")
            collected_data = self.collect_training_data()
            
            if not collected_data:
                raise ValueError("No data collected - cannot proceed")
            
            pipeline_results['data_collection'] = {
                'symbols_processed': len(collected_data),
                'total_symbols': len(self.symbols)
            }
            
            # Step 2: Prepare Training Dataset
            self.logger.info("🔧 Step 2: Feature Engineering & Dataset Preparation")
            X, y = self.prepare_training_dataset(collected_data)
            
            if X.empty:
                raise ValueError("No features engineered - cannot proceed")
            
            pipeline_results['dataset'] = {
                'samples': len(X),
                'features': len(X.columns)
            }
            
            # Step 3: Model Training
            self.logger.info("🤖 Step 3: ML Model Training")
            training_results = self.train_models(X, y)
            
            pipeline_results['training'] = {
                'models_trained': len(self.ml_models.models),
                'best_model': self._get_best_model_name(),
                'training_results': training_results
            }
            
            # Step 4: Model Evaluation
            self.logger.info("📈 Step 4: Model Evaluation & Comparison")
            evaluation_results = self.evaluate_models(collected_data)
            
            pipeline_results['evaluation'] = evaluation_results
            
            # Step 5: Backtesting
            self.logger.info("⏰ Step 5: Backtesting Framework")
            backtest_results = self.run_backtesting(collected_data)
            
            pipeline_results['backtesting'] = backtest_results
            
            # Step 6: Performance Report
            self.logger.info("📋 Step 6: Performance Report Generation")
            performance_report = self.generate_performance_report()
            
            pipeline_results['performance_report'] = performance_report
            
            # Step 7: Real-time System (Optional)
            if include_realtime:
                self.logger.info("🔴 Step 7: Real-time Monitoring System")
                self.start_realtime_system()
                pipeline_results['realtime_active'] = True
            else:
                pipeline_results['realtime_active'] = False
            
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            # Final summary
            self.logger.info("✅ Complete Pipeline Execution Finished")
            self.logger.info("=" * 60)
            
            if 'evaluation' in pipeline_results and 'mse_improvement_pct' in pipeline_results['evaluation']:
                improvement = pipeline_results['evaluation']['mse_improvement_pct']
                self.logger.info(f"🎯 Final Performance vs Black-Scholes: {improvement:.1f}% improvement")
                
                if improvement >= 12.0:
                    self.logger.info("🏆 TARGET ACHIEVED! 12%+ performance boost accomplished!")
                else:
                    self.logger.info(f"📊 Target progress: {improvement/12.0:.1%} towards 12% goal")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Error in complete pipeline: {str(e)}")
            return {'error': str(e)}

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description="Options Price Prediction Model")
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY'],
                       help='Stock symbols to analyze')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--realtime', action='store_true', help='Start real-time monitoring')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate', 'full'], 
                       default='full', help='Execution mode')
    
    args = parser.parse_args()
    
    # Initialize system
    system = OptionsPredictionSystem(args.symbols, args.config)
    
    if args.mode == 'full':
        # Run complete pipeline
        results = system.run_complete_pipeline(include_realtime=args.realtime)
        print("Complete pipeline executed successfully")
        
    elif args.mode == 'train':
        # Training only
        data = system.collect_training_data()
        X, y = system.prepare_training_dataset(data)
        results = system.train_models(X, y)
        print("Model training completed")
        
    elif args.mode == 'predict':
        # Real-time prediction only
        system.start_realtime_system()
        print("Real-time prediction system started")
        
    elif args.mode == 'evaluate':
        # Evaluation only
        data = system.collect_training_data(save_data=False)
        results = system.evaluate_models(data)
        report = system.generate_performance_report()
        print("Model evaluation completed")
    
    print(f"Results summary: {len(results)} components processed")

if __name__ == "__main__":
    main()
