"""
Real-time Options Price Prediction System

This module provides near real-time prediction capabilities including:
- Live data fetching from yFinance
- Real-time feature engineering
- Model prediction pipeline
- Signal generation
- Risk monitoring
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import threading
import queue
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Callable
import json
from dataclasses import dataclass

from data_collector import OptionsDataCollector
from feature_engineering import OptionsFeatureEngineering
from ml_models import OptionsMLModels
from black_scholes import BlackScholesModel

warnings.filterwarnings('ignore')

@dataclass
class PredictionSignal:
    """Data class for prediction signals"""
    symbol: str
    timestamp: datetime
    option_type: str
    strike: float
    expiration: str
    current_price: float
    predicted_price: float
    bs_price: float
    confidence: float
    signal_strength: str
    recommendation: str

class RealTimePredictionEngine:
    """
    Real-time prediction engine for options
    """
    
    def __init__(self, symbols: List[str], models_path: str = "../models/"):
        """
        Initialize real-time prediction engine
        
        Args:
            symbols: List of symbols to monitor
            models_path: Path to trained models
        """
        self.symbols = symbols
        self.models_path = models_path
        self.is_running = False
        self.prediction_queue = queue.Queue()
        self.signal_callbacks = []
        
        # Initialize components
        self.data_collector = OptionsDataCollector(symbols)
        self.feature_engineer = OptionsFeatureEngineering()
        self.ml_models = OptionsMLModels()
        self.bs_model = BlackScholesModel()
        
        self.logger = self._setup_logging()
        
        # Load trained models
        self._load_models()
        
        # Cache for recent data
        self.data_cache = {}
        self.last_update = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger
    
    def _load_models(self):
        """Load pre-trained ML models"""
        try:
            self.ml_models.load_models(self.models_path)
            self.logger.info(f"Loaded {len(self.ml_models.models)} trained models")
        except Exception as e:
            self.logger.warning(f"Could not load models: {str(e)}. Will use Black-Scholes only.")
    
    def add_signal_callback(self, callback: Callable[[PredictionSignal], None]):
        """
        Add callback function for prediction signals
        
        Args:
            callback: Function to call when new signal is generated
        """
        self.signal_callbacks.append(callback)
    
    def fetch_live_data(self, symbol: str) -> Dict:
        """
        Fetch live option chain and price data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with live data
        """
        try:
            # Check cache age
            cache_key = f"{symbol}_live"
            current_time = datetime.now()
            
            if (cache_key in self.data_cache and 
                cache_key in self.last_update and
                (current_time - self.last_update[cache_key]).seconds < 60):  # Cache for 1 minute
                return self.data_cache[cache_key]
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            
            # Current price
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                return {}
                
            current_price = hist['Close'].iloc[-1]
            
            # Option chains (limit to first 3 expirations for speed)
            expirations = ticker.options[:3] if ticker.options else []
            
            live_data = {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': current_time,
                'option_chains': {}
            }
            
            for exp_date in expirations:
                try:
                    chain = ticker.option_chain(exp_date)
                    
                    # Process calls and puts
                    for option_type, options in [('calls', chain.calls), ('puts', chain.puts)]:
                        if not options.empty:
                            options_df = options.copy()
                            options_df['type'] = 'call' if option_type == 'calls' else 'put'
                            options_df['symbol'] = symbol
                            options_df['current_price'] = current_price
                            options_df['expiration'] = exp_date
                            options_df['time_to_expiry'] = self._calculate_time_to_expiry(exp_date)
                            options_df['moneyness'] = options_df['strike'] / current_price
                            options_df['mid_price'] = (options_df['bid'] + options_df['ask']) / 2
                            
                            # Filter for liquid options
                            liquid_options = options_df[
                                (options_df['volume'] > 0) & 
                                (options_df['bid'] > 0) & 
                                (options_df['ask'] > 0) &
                                (options_df['moneyness'] >= 0.8) &
                                (options_df['moneyness'] <= 1.2)
                            ]
                            
                            if not liquid_options.empty:
                                live_data['option_chains'][f"{exp_date}_{option_type}"] = liquid_options
                                
                except Exception as exp_error:
                    self.logger.warning(f"Error fetching {exp_date} options for {symbol}: {str(exp_error)}")
                    continue
            
            # Update cache
            self.data_cache[cache_key] = live_data
            self.last_update[cache_key] = current_time
            
            return live_data
            
        except Exception as e:
            self.logger.error(f"Error fetching live data for {symbol}: {str(e)}")
            return {}
    
    def generate_predictions(self, symbol: str) -> List[PredictionSignal]:
        """
        Generate predictions for all liquid options of a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of prediction signals
        """
        try:
            signals = []
            
            # Get live data
            live_data = self.fetch_live_data(symbol)
            if not live_data or 'option_chains' not in live_data:
                return signals
            
            # Get historical data for feature engineering
            historical_data = self.data_collector.get_historical_data(symbol, period="1y")
            if historical_data.empty:
                self.logger.warning(f"No historical data for {symbol}")
                return signals
            
            # Get sentiment data
            sentiment_data = self.data_collector.get_market_sentiment_data(symbol)
            
            # Process each option chain
            for chain_key, option_df in live_data['option_chains'].items():
                try:
                    if option_df.empty:
                        continue
                    
                    # Engineer features
                    engineered_df = self.feature_engineer.engineer_all_features(
                        historical_data, option_df, sentiment_data
                    )
                    
                    # Prepare features for prediction
                    feature_cols = [col for col in engineered_df.columns 
                                  if col not in ['symbol', 'contractSymbol', 'expiration', 
                                               'type', 'strike', 'lastTradeDate']]
                    
                    X_features = engineered_df[feature_cols].select_dtypes(include=[np.number])
                    X_features = X_features.fillna(X_features.median())
                    
                    # Generate ML predictions if models are available
                    ml_predictions = None
                    if self.ml_models.models:
                        try:
                            # Use the best performing model
                            ml_predictions = self.ml_models.predict(X_features, 'best')
                        except Exception as ml_error:
                            self.logger.warning(f"ML prediction failed: {str(ml_error)}")
                    
                    # Generate Black-Scholes baseline predictions
                    bs_predictions = self.bs_model.batch_option_pricing(engineered_df)
                    
                    # Create signals for each option
                    for idx, row in engineered_df.iterrows():
                        try:
                            # ML prediction
                            ml_price = ml_predictions[idx] if ml_predictions is not None else None
                            bs_price = bs_predictions.loc[idx, 'bs_price']
                            current_market_price = row['mid_price']
                            
                            # Calculate prediction confidence
                            if ml_price is not None:
                                predicted_price = ml_price
                                # Confidence based on agreement with BS model
                                price_diff = abs(ml_price - bs_price) / bs_price
                                confidence = max(0, 1 - price_diff)
                            else:
                                predicted_price = bs_price
                                confidence = 0.5  # Medium confidence for BS only
                            
                            # Generate signal strength and recommendation
                            price_deviation = (predicted_price - current_market_price) / current_market_price
                            
                            if abs(price_deviation) > 0.05:  # 5% threshold
                                signal_strength = 'strong'
                            elif abs(price_deviation) > 0.02:  # 2% threshold
                                signal_strength = 'moderate'
                            else:
                                signal_strength = 'weak'
                            
                            if price_deviation > 0.02:
                                recommendation = 'BUY' if row['type'] == 'call' else 'SELL PUT'
                            elif price_deviation < -0.02:
                                recommendation = 'SELL' if row['type'] == 'call' else 'BUY PUT'
                            else:
                                recommendation = 'HOLD'
                            
                            # Create signal
                            signal = PredictionSignal(
                                symbol=symbol,
                                timestamp=live_data['timestamp'],
                                option_type=row['type'],
                                strike=row['strike'],
                                expiration=row['expiration'],
                                current_price=current_market_price,
                                predicted_price=predicted_price,
                                bs_price=bs_price,
                                confidence=confidence,
                                signal_strength=signal_strength,
                                recommendation=recommendation
                            )
                            
                            signals.append(signal)
                            
                        except Exception as signal_error:
                            self.logger.warning(f"Error creating signal for option {idx}: {str(signal_error)}")
                            continue
                    
                except Exception as chain_error:
                    self.logger.warning(f"Error processing chain {chain_key}: {str(chain_error)}")
                    continue
            
            # Filter for significant signals only
            significant_signals = [s for s in signals if s.signal_strength in ['moderate', 'strong']]
            
            self.logger.info(f"Generated {len(significant_signals)} significant signals for {symbol}")
            return significant_signals
            
        except Exception as e:
            self.logger.error(f"Error generating predictions for {symbol}: {str(e)}")
            return []
    
    def start_realtime_monitoring(self, update_interval: int = 300):
        """
        Start real-time monitoring and prediction
        
        Args:
            update_interval: Update interval in seconds
        """
        def monitoring_loop():
            """Main monitoring loop"""
            while self.is_running:
                try:
                    for symbol in self.symbols:
                        # Generate predictions
                        signals = self.generate_predictions(symbol)
                        
                        # Add significant signals to queue and trigger callbacks
                        for signal in signals:
                            if signal.signal_strength in ['moderate', 'strong']:
                                self.prediction_queue.put(signal)
                                
                                # Trigger callbacks
                                for callback in self.signal_callbacks:
                                    try:
                                        callback(signal)
                                    except Exception as cb_error:
                                        self.logger.warning(f"Callback error: {str(cb_error)}")
                    
                    # Wait for next update
                    time.sleep(update_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        if not self.is_running:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info(f"Started real-time monitoring for {len(self.symbols)} symbols")
        else:
            self.logger.warning("Real-time monitoring is already running")
    
    def stop_realtime_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=10)
        self.logger.info("Stopped real-time monitoring")
    
    def get_latest_signals(self, max_signals: int = 10) -> List[PredictionSignal]:
        """
        Get latest prediction signals from queue
        
        Args:
            max_signals: Maximum number of signals to return
            
        Returns:
            List of latest signals
        """
        signals = []
        count = 0
        
        while not self.prediction_queue.empty() and count < max_signals:
            try:
                signal = self.prediction_queue.get_nowait()
                signals.append(signal)
                count += 1
            except queue.Empty:
                break
        
        return signals
    
    def calculate_portfolio_signals(self, positions: pd.DataFrame) -> Dict:
        """
        Calculate portfolio-level signals and risk metrics
        
        Args:
            positions: DataFrame with current portfolio positions
            
        Returns:
            Dictionary with portfolio signals
        """
        try:
            portfolio_signals = {
                'timestamp': datetime.now(),
                'total_positions': len(positions),
                'portfolio_value': 0,
                'portfolio_greeks': {},
                'risk_metrics': {},
                'alerts': []
            }
            
            if positions.empty:
                return portfolio_signals
            
            # Calculate portfolio value and Greeks
            total_value = 0
            portfolio_delta = 0
            portfolio_gamma = 0
            portfolio_theta = 0
            portfolio_vega = 0
            
            for _, position in positions.iterrows():
                try:
                    symbol = position['symbol']
                    quantity = position['quantity']
                    
                    # Get current option data
                    live_data = self.fetch_live_data(symbol)
                    if not live_data:
                        continue
                    
                    # Find matching option in live data
                    option_found = False
                    for chain_key, option_df in live_data['option_chains'].items():
                        match = option_df[
                            (option_df['strike'] == position['strike']) &
                            (option_df['type'] == position['option_type'])
                        ]
                        
                        if not match.empty:
                            option_data = match.iloc[0]
                            current_price = option_data['mid_price']
                            
                            # Calculate Greeks
                            greeks = self.bs_model.calculate_greeks(
                                live_data['current_price'],
                                position['strike'],
                                self._calculate_time_to_expiry(position['expiration']),
                                option_data['impliedVolatility'],
                                position['option_type']
                            )
                            
                            # Update portfolio metrics
                            position_value = quantity * current_price * 100  # Options are per 100 shares
                            total_value += position_value
                            
                            portfolio_delta += quantity * greeks.get('delta', 0) * 100
                            portfolio_gamma += quantity * greeks.get('gamma', 0) * 100
                            portfolio_theta += quantity * greeks.get('theta', 0) * 100
                            portfolio_vega += quantity * greeks.get('vega', 0) * 100
                            
                            option_found = True
                            break
                    
                    if not option_found:
                        portfolio_signals['alerts'].append(
                            f"Could not find current price for {symbol} {position['strike']} {position['option_type']}"
                        )
                        
                except Exception as pos_error:
                    self.logger.warning(f"Error processing position: {str(pos_error)}")
                    continue
            
            portfolio_signals['portfolio_value'] = total_value
            portfolio_signals['portfolio_greeks'] = {
                'delta': portfolio_delta,
                'gamma': portfolio_gamma,
                'theta': portfolio_theta,
                'vega': portfolio_vega
            }
            
            # Risk alerts
            if abs(portfolio_delta) > 1000:
                portfolio_signals['alerts'].append(f"High portfolio delta: {portfolio_delta:.0f}")
            
            if abs(portfolio_gamma) > 500:
                portfolio_signals['alerts'].append(f"High portfolio gamma: {portfolio_gamma:.0f}")
            
            if portfolio_theta < -500:
                portfolio_signals['alerts'].append(f"High time decay risk: {portfolio_theta:.0f}")
            
            return portfolio_signals
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio signals: {str(e)}")
            return {}
    
    def _calculate_time_to_expiry(self, expiration_date: str) -> float:
        """Calculate time to expiry in years"""
        try:
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            today = datetime.now()
            time_diff = (exp_date - today).days
            return max(time_diff / 365.0, 0.001)
        except:
            return 0.001
    
    def run_single_prediction(self, symbol: str, strike: float, 
                            expiration: str, option_type: str) -> Optional[PredictionSignal]:
        """
        Run prediction for a single option
        
        Args:
            symbol: Stock symbol
            strike: Strike price
            expiration: Expiration date
            option_type: 'call' or 'put'
            
        Returns:
            Prediction signal or None
        """
        try:
            # Get live data
            live_data = self.fetch_live_data(symbol)
            if not live_data:
                return None
            
            # Find the specific option
            option_data = None
            for chain_key, options_df in live_data['option_chains'].items():
                match = options_df[
                    (options_df['strike'] == strike) &
                    (options_df['type'] == option_type) &
                    (options_df['expiration'] == expiration)
                ]
                
                if not match.empty:
                    option_data = match.iloc[0]
                    break
            
            if option_data is None:
                self.logger.warning(f"Option not found: {symbol} {strike} {expiration} {option_type}")
                return None
            
            # Generate prediction
            signals = self.generate_predictions(symbol)
            
            # Find matching signal
            for signal in signals:
                if (signal.strike == strike and 
                    signal.expiration == expiration and 
                    signal.option_type == option_type):
                    return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}")
            return None

class OptionsSignalProcessor:
    """
    Process and analyze prediction signals
    """
    
    def __init__(self):
        self.signal_history = []
        self.logger = logging.getLogger(__name__)
    
    def process_signal(self, signal: PredictionSignal) -> Dict:
        """
        Process incoming prediction signal
        
        Args:
            signal: Prediction signal
            
        Returns:
            Dictionary with processed signal information
        """
        try:
            # Add to history
            self.signal_history.append(signal)
            
            # Keep only recent signals (last 1000)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            # Signal processing logic
            processed = {
                'signal_id': f"{signal.symbol}_{signal.strike}_{signal.expiration}_{signal.option_type}",
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'recommendation': signal.recommendation,
                'confidence': signal.confidence,
                'price_prediction': signal.predicted_price,
                'current_price': signal.current_price,
                'expected_return': (signal.predicted_price - signal.current_price) / signal.current_price,
                'risk_level': self._calculate_risk_level(signal),
                'priority': self._calculate_priority(signal)
            }
            
            # Log significant signals
            if signal.signal_strength in ['moderate', 'strong']:
                self.logger.info(f"Signal: {signal.symbol} {signal.strike} {signal.option_type} "
                               f"- {signal.recommendation} (Confidence: {signal.confidence:.2f})")
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
            return {}
    
    def _calculate_risk_level(self, signal: PredictionSignal) -> str:
        """Calculate risk level for signal"""
        try:
            # Risk factors
            time_risk = 1.0 if signal.expiration else 0.0  # Placeholder
            moneyness_risk = abs(signal.strike / signal.current_price - 1.0)
            confidence_risk = 1.0 - signal.confidence
            
            total_risk = (time_risk + moneyness_risk + confidence_risk) / 3
            
            if total_risk > 0.7:
                return 'HIGH'
            elif total_risk > 0.4:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except:
            return 'MEDIUM'
    
    def _calculate_priority(self, signal: PredictionSignal) -> int:
        """Calculate signal priority (1-10, 10 = highest)"""
        try:
            priority = 5  # Base priority
            
            # Adjust for signal strength
            if signal.signal_strength == 'strong':
                priority += 3
            elif signal.signal_strength == 'moderate':
                priority += 1
            
            # Adjust for confidence
            if signal.confidence > 0.8:
                priority += 2
            elif signal.confidence > 0.6:
                priority += 1
            
            # Adjust for expected return
            expected_return = abs((signal.predicted_price - signal.current_price) / signal.current_price)
            if expected_return > 0.1:
                priority += 2
            elif expected_return > 0.05:
                priority += 1
            
            return min(max(priority, 1), 10)
            
        except:
            return 5

    def get_signal_summary(self, hours_back: int = 1) -> Dict:
        """
        Get summary of recent signals
        
        Args:
            hours_back: Hours to look back
            
        Returns:
            Dictionary with signal summary
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_signals = [s for s in self.signal_history if s.timestamp > cutoff_time]
            
            if not recent_signals:
                return {'total_signals': 0}
            
            summary = {
                'total_signals': len(recent_signals),
                'by_recommendation': {},
                'by_strength': {},
                'by_symbol': {},
                'average_confidence': np.mean([s.confidence for s in recent_signals]),
                'high_priority_signals': len([s for s in recent_signals if s.signal_strength == 'strong'])
            }
            
            # Count by categories
            for signal in recent_signals:
                # By recommendation
                rec = signal.recommendation
                summary['by_recommendation'][rec] = summary['by_recommendation'].get(rec, 0) + 1
                
                # By strength
                strength = signal.signal_strength
                summary['by_strength'][strength] = summary['by_strength'].get(strength, 0) + 1
                
                # By symbol
                symbol = signal.symbol
                summary['by_symbol'][symbol] = summary['by_symbol'].get(symbol, 0) + 1
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating signal summary: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    print("Real-time Prediction Engine initialized successfully")
    
    # Initialize engine
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    engine = RealTimePredictionEngine(symbols)
    
    # Test single prediction
    try:
        signal = engine.run_single_prediction('AAPL', 150.0, '2024-01-19', 'call')
        if signal:
            print(f"Generated signal: {signal.recommendation} for AAPL 150 call")
            print(f"Confidence: {signal.confidence:.2f}, Expected price: ${signal.predicted_price:.2f}")
        else:
            print("No signal generated (option may not be liquid)")
    except Exception as e:
        print(f"Error in example: {e}")
    
    # Test batch predictions
    try:
        all_signals = engine.generate_predictions('AAPL')
        print(f"Generated {len(all_signals)} signals for AAPL")
        
        strong_signals = [s for s in all_signals if s.signal_strength == 'strong']
        if strong_signals:
            print(f"Found {len(strong_signals)} strong signals:")
            for signal in strong_signals[:3]:  # Show first 3
                print(f"  {signal.strike} {signal.option_type} - {signal.recommendation}")
    except Exception as e:
        print(f"Error in batch prediction example: {e}")
