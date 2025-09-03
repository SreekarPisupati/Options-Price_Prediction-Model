"""
Data Collection Module for Options Price Prediction

This module handles data collection from various sources including:
- yFinance for stock prices and option chains
- Volatility calculations
- Sentiment data integration
- Real-time data fetching capabilities
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class OptionsDataCollector:
    """
    Comprehensive data collector for options trading data
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize the data collector
        
        Args:
            symbols: List of stock symbols to track
        """
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the data collector"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger
    
    def get_historical_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        Get historical stock price data
        
        Args:
            symbol: Stock symbol
            period: Time period (1y, 2y, 5y, etc.)
            
        Returns:
            DataFrame with historical price data
        """
        try:
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period=period)
            hist_data.reset_index(inplace=True)
            
            # Calculate additional features
            hist_data['Returns'] = hist_data['Close'].pct_change()
            hist_data['Log_Returns'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
            hist_data['Volatility_10'] = hist_data['Returns'].rolling(window=10).std() * np.sqrt(252)
            hist_data['Volatility_30'] = hist_data['Returns'].rolling(window=30).std() * np.sqrt(252)
            hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
            hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
            hist_data['RSI'] = self._calculate_rsi(hist_data['Close'])
            hist_data['MACD'] = self._calculate_macd(hist_data['Close'])
            
            self.logger.info(f"Successfully fetched historical data for {symbol}")
            return hist_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_option_chain(self, symbol: str, expiration_date: str = None) -> Dict:
        """
        Get option chain data for a given symbol
        
        Args:
            symbol: Stock symbol
            expiration_date: Specific expiration date (YYYY-MM-DD)
            
        Returns:
            Dictionary containing calls and puts data
        """
        try:
            stock = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = stock.options
            if not expirations:
                self.logger.warning(f"No options available for {symbol}")
                return {}
            
            # Use the nearest expiration if none specified
            if expiration_date is None:
                expiration_date = expirations[0]
            
            # Get option chain
            option_chain = stock.option_chain(expiration_date)
            
            # Process calls
            calls_df = option_chain.calls.copy()
            calls_df['type'] = 'call'
            calls_df['symbol'] = symbol
            calls_df['expiration'] = expiration_date
            
            # Process puts
            puts_df = option_chain.puts.copy()
            puts_df['type'] = 'put'
            puts_df['symbol'] = symbol
            puts_df['expiration'] = expiration_date
            
            # Calculate additional features
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            
            for df in [calls_df, puts_df]:
                df['moneyness'] = df['strike'] / current_price
                df['time_to_expiry'] = self._calculate_time_to_expiry(expiration_date)
                df['mid_price'] = (df['bid'] + df['ask']) / 2
                df['spread'] = df['ask'] - df['bid']
                df['spread_pct'] = df['spread'] / df['mid_price']
                
            self.logger.info(f"Successfully fetched option chain for {symbol}, expiration: {expiration_date}")
            
            return {
                'calls': calls_df,
                'puts': puts_df,
                'current_price': current_price,
                'expiration_date': expiration_date
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching option chain for {symbol}: {str(e)}")
            return {}
    
    def get_all_option_expirations(self, symbol: str) -> Dict:
        """
        Get option chains for all available expiration dates
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with all expiration dates and their option chains
        """
        try:
            stock = yf.Ticker(symbol)
            expirations = stock.options
            
            all_chains = {}
            for exp_date in expirations[:6]:  # Limit to first 6 expirations
                chain_data = self.get_option_chain(symbol, exp_date)
                if chain_data:
                    all_chains[exp_date] = chain_data
                    
            return all_chains
            
        except Exception as e:
            self.logger.error(f"Error fetching all option expirations for {symbol}: {str(e)}")
            return {}
    
    def calculate_implied_volatility_features(self, option_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate implied volatility based features
        
        Args:
            option_data: Option chain DataFrame
            
        Returns:
            DataFrame with additional IV features
        """
        try:
            # IV term structure
            option_data['iv_rank'] = option_data['impliedVolatility'].rank(pct=True)
            option_data['iv_zscore'] = (option_data['impliedVolatility'] - 
                                       option_data['impliedVolatility'].mean()) / option_data['impliedVolatility'].std()
            
            # Volume/OI ratios
            option_data['volume_oi_ratio'] = option_data['volume'] / (option_data['openInterest'] + 1)
            option_data['volume_weight'] = option_data['volume'] / option_data['volume'].sum()
            
            return option_data
            
        except Exception as e:
            self.logger.error(f"Error calculating IV features: {str(e)}")
            return option_data
    
    def get_market_sentiment_data(self, symbol: str) -> Dict:
        """
        Get market sentiment indicators (placeholder for various sentiment sources)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with sentiment indicators
        """
        try:
            # Placeholder for sentiment data - in a real implementation, 
            # this would connect to news APIs, social media APIs, etc.
            
            # VIX data for market sentiment
            if symbol in ['SPY', 'SPX']:
                vix = yf.Ticker('^VIX')
                vix_data = vix.history(period="30d")
                vix_current = vix_data['Close'].iloc[-1]
                vix_percentile = (vix_current - vix_data['Close'].min()) / (vix_data['Close'].max() - vix_data['Close'].min())
                
                return {
                    'vix_level': vix_current,
                    'vix_percentile': vix_percentile,
                    'market_fear': 'high' if vix_current > 25 else 'medium' if vix_current > 20 else 'low'
                }
            
            # For individual stocks, create proxy sentiment metrics
            stock = yf.Ticker(symbol)
            info = stock.info
            
            sentiment_score = 0.5  # Neutral baseline
            
            # Volume-based sentiment
            recent_data = stock.history(period="10d")
            avg_volume = recent_data['Volume'].mean()
            recent_volume = recent_data['Volume'].iloc[-1]
            volume_ratio = recent_volume / avg_volume
            
            # Price momentum sentiment
            price_change_5d = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-5]) / recent_data['Close'].iloc[-5]
            
            return {
                'volume_sentiment': min(max(volume_ratio / 2, 0), 1),
                'momentum_sentiment': min(max((price_change_5d + 0.1) / 0.2, 0), 1),
                'overall_sentiment': sentiment_score
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching sentiment data for {symbol}: {str(e)}")
            return {'overall_sentiment': 0.5}
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_time_to_expiry(self, expiration_date: str) -> float:
        """Calculate time to expiry in years"""
        try:
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            today = datetime.now()
            time_diff = (exp_date - today).days
            return max(time_diff / 365.0, 0.001)  # Minimum 1 day
        except:
            return 0.001
    
    def collect_comprehensive_data(self, symbol: str, save_to_file: bool = True) -> Dict:
        """
        Collect comprehensive data for a symbol including historical prices, 
        option chains, and sentiment data
        
        Args:
            symbol: Stock symbol
            save_to_file: Whether to save data to files
            
        Returns:
            Dictionary with all collected data
        """
        try:
            self.logger.info(f"Starting comprehensive data collection for {symbol}")
            
            # Collect all data
            historical_data = self.get_historical_data(symbol)
            option_chains = self.get_all_option_expirations(symbol)
            sentiment_data = self.get_market_sentiment_data(symbol)
            
            # Combine all data
            comprehensive_data = {
                'symbol': symbol,
                'historical_data': historical_data,
                'option_chains': option_chains,
                'sentiment_data': sentiment_data,
                'collection_timestamp': datetime.now().isoformat()
            }
            
            # Save to file if requested
            if save_to_file:
                self._save_data_to_file(comprehensive_data, symbol)
            
            self.logger.info(f"Completed comprehensive data collection for {symbol}")
            return comprehensive_data
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive data collection for {symbol}: {str(e)}")
            return {}
    
    def _save_data_to_file(self, data: Dict, symbol: str):
        """Save collected data to files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save historical data
            if not data['historical_data'].empty:
                hist_filename = f"../data/{symbol}_historical_{timestamp}.csv"
                data['historical_data'].to_csv(hist_filename, index=False)
            
            # Save option chains
            for exp_date, chain_data in data['option_chains'].items():
                if 'calls' in chain_data and not chain_data['calls'].empty:
                    calls_filename = f"../data/{symbol}_calls_{exp_date}_{timestamp}.csv"
                    chain_data['calls'].to_csv(calls_filename, index=False)
                
                if 'puts' in chain_data and not chain_data['puts'].empty:
                    puts_filename = f"../data/{symbol}_puts_{exp_date}_{timestamp}.csv"
                    chain_data['puts'].to_csv(puts_filename, index=False)
                    
        except Exception as e:
            self.logger.error(f"Error saving data to file: {str(e)}")

if __name__ == "__main__":
    # Example usage
    collector = OptionsDataCollector(['AAPL', 'GOOGL', 'MSFT'])
    
    # Collect data for Apple
    apple_data = collector.collect_comprehensive_data('AAPL')
    print(f"Collected data for AAPL: {len(apple_data.get('option_chains', {}))} expiration dates")
