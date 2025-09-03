"""
Feature Engineering Module for Options Price Prediction

This module handles comprehensive feature engineering including:
- Volatility features (historical, implied, GARCH)
- Technical indicators
- Greek calculations
- Time-based features
- Sentiment features
- Market regime features
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import logging
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

class OptionsFeatureEngineering:
    """
    Comprehensive feature engineering for options trading models
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize feature engineering pipeline
        
        Args:
            risk_free_rate: Risk-free interest rate for Black-Scholes calculations
        """
        self.risk_free_rate = risk_free_rate
        self.scaler = None
        self.selected_features = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger
    
    def calculate_black_scholes_greeks(self, option_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Black-Scholes Greeks and theoretical prices
        
        Args:
            option_data: DataFrame with option chain data
            
        Returns:
            DataFrame with added Greeks columns
        """
        try:
            df = option_data.copy()
            
            # Required parameters
            S = df['current_price']  # Current stock price
            K = df['strike']         # Strike price
            T = df['time_to_expiry'] # Time to expiration
            r = self.risk_free_rate  # Risk-free rate
            sigma = df['impliedVolatility']  # Implied volatility
            
            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Black-Scholes theoretical prices
            if 'type' in df.columns:
                call_mask = df['type'] == 'call'
                put_mask = df['type'] == 'put'
                
                df.loc[call_mask, 'bs_price'] = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
                df.loc[put_mask, 'bs_price'] = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                # Assume all are calls if type not specified
                df['bs_price'] = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            
            # Calculate Greeks
            df['delta'] = norm.cdf(d1)
            df['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            df['theta'] = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                          r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
            df['vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100
            df['rho'] = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
            
            # Adjust Greeks for puts
            if 'type' in df.columns:
                df.loc[put_mask, 'delta'] = df.loc[put_mask, 'delta'] - 1
                df.loc[put_mask, 'theta'] = df.loc[put_mask, 'theta'] + r * K * np.exp(-r*T) / 365
                df.loc[put_mask, 'rho'] = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
            
            # Price deviation from theoretical
            if 'mid_price' in df.columns:
                df['price_deviation'] = (df['mid_price'] - df['bs_price']) / df['bs_price']
                df['price_deviation_abs'] = np.abs(df['price_deviation'])
            
            self.logger.info("Successfully calculated Black-Scholes Greeks")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {str(e)}")
            return option_data
    
    def calculate_volatility_features(self, historical_data: pd.DataFrame, 
                                    option_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate comprehensive volatility features
        
        Args:
            historical_data: Historical price data
            option_data: Option chain data (optional)
            
        Returns:
            DataFrame with volatility features
        """
        try:
            df = historical_data.copy()
            
            # Ensure we have returns
            if 'Returns' not in df.columns:
                df['Returns'] = df['Close'].pct_change()
            
            # Rolling volatilities
            windows = [5, 10, 20, 30, 60, 90]
            for window in windows:
                df[f'rv_{window}d'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
                df[f'rv_{window}d_rank'] = df[f'rv_{window}d'].rolling(window=252).rank(pct=True)
            
            # EWMA volatility
            df['ewma_vol'] = df['Returns'].ewm(span=30).std() * np.sqrt(252)
            
            # Garman-Klass volatility (if OHLC data available)
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                df['gk_vol'] = np.sqrt(
                    0.5 * np.log(df['High'] / df['Low'])**2 - 
                    (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open'])**2
                ) * np.sqrt(252)
            
            # Volatility clustering
            df['vol_cluster'] = (df['Returns'].abs() > df['Returns'].abs().rolling(20).mean()).astype(int)
            
            # Volatility regime
            short_vol = df['Returns'].rolling(10).std()
            long_vol = df['Returns'].rolling(60).std()
            df['vol_regime'] = (short_vol > long_vol).astype(int)
            
            # Jump detection
            threshold = df['Returns'].rolling(60).std() * 3
            df['jump'] = (df['Returns'].abs() > threshold).astype(int)
            df['jump_intensity'] = df['jump'].rolling(20).sum()
            
            # Volatility term structure (if option data available)
            if option_data is not None and 'impliedVolatility' in option_data.columns:
                # IV percentile
                current_iv = option_data['impliedVolatility'].mean()
                historical_iv = option_data.groupby('expiration')['impliedVolatility'].mean()
                
                df['iv_current'] = current_iv
                df['iv_percentile'] = stats.percentileofscore(historical_iv, current_iv) / 100
                
                # IV term structure slope
                if len(historical_iv) > 1:
                    iv_slope = np.polyfit(range(len(historical_iv)), historical_iv.values, 1)[0]
                    df['iv_term_structure_slope'] = iv_slope
            
            self.logger.info("Successfully calculated volatility features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {str(e)}")
            return historical_data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            data: Price data DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            df = data.copy()
            
            # Moving averages
            periods = [5, 10, 20, 50, 100, 200]
            for period in periods:
                df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
                
                # Price relative to moving average
                df[f'price_to_sma_{period}'] = df['Close'] / df[f'sma_{period}']
                df[f'price_to_ema_{period}'] = df['Close'] / df[f'ema_{period}']
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # RSI
            df['rsi'] = self._calculate_rsi(df['Close'])
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Stochastic
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            df['williams_r'] = -100 * (high_max - df['Close']) / (high_max - low_min)
            
            # Momentum indicators
            df['momentum_1d'] = df['Close'].pct_change(1)
            df['momentum_5d'] = df['Close'].pct_change(5)
            df['momentum_10d'] = df['Close'].pct_change(10)
            df['momentum_20d'] = df['Close'].pct_change(20)
            
            # Rate of Change
            df['roc_10d'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
            
            # Average True Range
            df['atr'] = self._calculate_atr(df)
            
            # Volume indicators
            if 'Volume' in df.columns:
                df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
                df['price_volume'] = df['Close'] * df['Volume']
                
                # On Balance Volume
                df['obv'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
                
                # Volume Rate of Change
                df['vroc'] = df['Volume'].pct_change(10)
            
            self.logger.info("Successfully calculated technical indicators")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return data
    
    def calculate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time-based features
        
        Args:
            data: DataFrame with datetime index or Date column
            
        Returns:
            DataFrame with time features
        """
        try:
            df = data.copy()
            
            # Ensure we have a datetime column
            if 'Date' in df.columns:
                date_col = pd.to_datetime(df['Date'])
            elif isinstance(df.index, pd.DatetimeIndex):
                date_col = df.index
            else:
                self.logger.warning("No datetime column found")
                return df
            
            # Basic time features
            df['year'] = date_col.year
            df['month'] = date_col.month
            df['day'] = date_col.day
            df['dayofweek'] = date_col.dayofweek
            df['dayofyear'] = date_col.dayofyear
            df['quarter'] = date_col.quarter
            
            # Market timing features
            df['is_monday'] = (df['dayofweek'] == 0).astype(int)
            df['is_friday'] = (df['dayofweek'] == 4).astype(int)
            df['is_month_end'] = date_col.is_month_end.astype(int)
            df['is_month_start'] = date_col.is_month_start.astype(int)
            df['is_quarter_end'] = date_col.is_quarter_end.astype(int)
            
            # Cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            
            # Days until expiration (if expiration date available)
            if 'expiration' in df.columns:
                exp_dates = pd.to_datetime(df['expiration'])
                df['days_to_expiry'] = (exp_dates - date_col).dt.days
                df['time_to_expiry'] = df['days_to_expiry'] / 365.0
                
                # Expiration month effect
                df['expiry_week'] = (df['days_to_expiry'] <= 7).astype(int)
                df['expiry_month'] = (df['days_to_expiry'] <= 30).astype(int)
            
            self.logger.info("Successfully calculated time features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating time features: {str(e)}")
            return data
    
    def calculate_market_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market regime and macro features
        
        Args:
            data: Price data DataFrame
            
        Returns:
            DataFrame with regime features
        """
        try:
            df = data.copy()
            
            # Trend identification
            sma_short = df['Close'].rolling(window=20).mean()
            sma_long = df['Close'].rolling(window=50).mean()
            
            df['trend_bullish'] = (sma_short > sma_long).astype(int)
            df['trend_strength'] = (sma_short - sma_long) / sma_long
            
            # Volatility regime
            vol_short = df['Returns'].rolling(10).std() if 'Returns' in df.columns else df['Close'].pct_change().rolling(10).std()
            vol_long = df['Returns'].rolling(60).std() if 'Returns' in df.columns else df['Close'].pct_change().rolling(60).std()
            
            df['high_vol_regime'] = (vol_short > vol_long * 1.5).astype(int)
            
            # Price momentum regime
            returns_20d = df['Close'].pct_change(20)
            df['momentum_regime'] = np.where(returns_20d > 0.05, 1, 
                                           np.where(returns_20d < -0.05, -1, 0))
            
            # Market stress indicators
            drawdown = (df['Close'] / df['Close'].rolling(252).max() - 1)
            df['drawdown'] = drawdown
            df['stress_regime'] = (drawdown < -0.1).astype(int)
            
            # Volatility clustering
            if 'Returns' in df.columns:
                vol_proxy = df['Returns'].abs()
                df['vol_clustering'] = (vol_proxy > vol_proxy.rolling(20).mean() * 1.5).astype(int)
            
            self.logger.info("Successfully calculated market regime features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating market regime features: {str(e)}")
            return data
    
    def add_sentiment_features(self, data: pd.DataFrame, sentiment_data: Dict) -> pd.DataFrame:
        """
        Add sentiment features to the dataset
        
        Args:
            data: Main DataFrame
            sentiment_data: Dictionary with sentiment indicators
            
        Returns:
            DataFrame with sentiment features
        """
        try:
            df = data.copy()
            
            # Add sentiment scores
            for key, value in sentiment_data.items():
                if isinstance(value, (int, float)):
                    df[f'sentiment_{key}'] = value
                elif isinstance(value, str):
                    # Convert categorical sentiment to numerical
                    sentiment_mapping = {'high': 0.8, 'medium': 0.5, 'low': 0.2}
                    df[f'sentiment_{key}'] = sentiment_mapping.get(value, 0.5)
            
            # Create composite sentiment score
            sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
            if sentiment_cols:
                df['sentiment_composite'] = df[sentiment_cols].mean(axis=1)
                df['sentiment_extreme'] = (
                    (df['sentiment_composite'] > 0.7) | 
                    (df['sentiment_composite'] < 0.3)
                ).astype(int)
            
            self.logger.info("Successfully added sentiment features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding sentiment features: {str(e)}")
            return data
    
    def create_lagged_features(self, data: pd.DataFrame, columns: List[str], 
                             lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features for specified columns
        
        Args:
            data: Input DataFrame
            columns: List of column names to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        try:
            df = data.copy()
            
            for col in columns:
                if col in df.columns:
                    for lag in lags:
                        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                        
                        # Rolling statistics of lagged features
                        if lag == 1:
                            df[f'{col}_rolling_mean_5'] = df[col].rolling(5).mean()
                            df[f'{col}_rolling_std_5'] = df[col].rolling(5).std()
                            df[f'{col}_rolling_max_5'] = df[col].rolling(5).max()
                            df[f'{col}_rolling_min_5'] = df[col].rolling(5).min()
            
            self.logger.info(f"Successfully created lagged features for {len(columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating lagged features: {str(e)}")
            return data
    
    def engineer_all_features(self, historical_data: pd.DataFrame, 
                            option_data: pd.DataFrame,
                            sentiment_data: Dict = None) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            historical_data: Historical price data
            option_data: Option chain data
            sentiment_data: Sentiment indicators
            
        Returns:
            Fully engineered feature dataset
        """
        try:
            self.logger.info("Starting comprehensive feature engineering")
            
            # Calculate Black-Scholes Greeks
            option_df = self.calculate_black_scholes_greeks(option_data)
            
            # Calculate volatility features
            hist_df = self.calculate_volatility_features(historical_data, option_data)
            
            # Technical indicators
            hist_df = self.calculate_technical_indicators(hist_df)
            
            # Time features
            hist_df = self.calculate_time_features(hist_df)
            option_df = self.calculate_time_features(option_df)
            
            # Market regime features
            hist_df = self.calculate_market_regime_features(hist_df)
            
            # Add sentiment features
            if sentiment_data:
                hist_df = self.add_sentiment_features(hist_df, sentiment_data)
                option_df = self.add_sentiment_features(option_df, sentiment_data)
            
            # Merge historical and option features
            # Get the latest historical features
            latest_hist = hist_df.iloc[-1:].copy()
            
            # Add historical features to option data
            hist_feature_cols = [col for col in latest_hist.columns 
                               if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            for col in hist_feature_cols:
                if col in latest_hist.columns:
                    option_df[f'hist_{col}'] = latest_hist[col].iloc[0]
            
            # Create interaction features
            option_df = self._create_interaction_features(option_df)
            
            # Remove infinite and NaN values
            option_df = option_df.replace([np.inf, -np.inf], np.nan)
            option_df = option_df.ffill().bfill()
            
            self.logger.info(f"Feature engineering completed. Final shape: {option_df.shape}")
            return option_df
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive feature engineering: {str(e)}")
            return option_data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        try:
            df = data.copy()
            
            # Key interaction features for options
            if all(col in df.columns for col in ['delta', 'moneyness']):
                df['delta_moneyness'] = df['delta'] * df['moneyness']
            
            if all(col in df.columns for col in ['vega', 'impliedVolatility']):
                df['vega_iv'] = df['vega'] * df['impliedVolatility']
            
            if all(col in df.columns for col in ['theta', 'time_to_expiry']):
                df['theta_time'] = df['theta'] * df['time_to_expiry']
            
            if all(col in df.columns for col in ['gamma', 'volume']):
                df['gamma_volume'] = df['gamma'] * np.log1p(df['volume'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating interaction features: {str(e)}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(window=window).mean()
        return atr
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       k: int = 50, method: str = 'f_regression') -> pd.DataFrame:
        """
        Select top k features using statistical methods
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            method: Feature selection method
            
        Returns:
            DataFrame with selected features
        """
        try:
            # Remove non-numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_cols]
            
            # Handle infinite and NaN values
            X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
            X_numeric = X_numeric.fillna(X_numeric.median())
            
            # Feature selection
            if method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=min(k, len(numeric_cols)))
            else:
                selector = SelectKBest(k=min(k, len(numeric_cols)))
            
            X_selected = selector.fit_transform(X_numeric, y)
            selected_features = numeric_cols[selector.get_support()]
            
            self.selected_features = selected_features
            result_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            self.logger.info(f"Selected {len(selected_features)} features out of {len(numeric_cols)}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            return X
    
    def scale_features(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale features using specified method
        
        Args:
            X: Feature matrix
            method: Scaling method ('standard', 'robust')
            
        Returns:
            Scaled feature matrix
        """
        try:
            if method == 'standard':
                if self.scaler is None:
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                else:
                    X_scaled = self.scaler.transform(X)
                    
            elif method == 'robust':
                if self.scaler is None:
                    self.scaler = RobustScaler()
                    X_scaled = self.scaler.fit_transform(X)
                else:
                    X_scaled = self.scaler.transform(X)
            else:
                self.logger.warning(f"Unknown scaling method: {method}")
                return X
            
            scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            self.logger.info(f"Successfully scaled features using {method} method")
            return scaled_df
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            return X

if __name__ == "__main__":
    # Example usage
    from data_collector import OptionsDataCollector
    
    collector = OptionsDataCollector()
    data = collector.collect_comprehensive_data('AAPL', save_to_file=False)
    
    engineer = OptionsFeatureEngineering()
    
    if data and 'historical_data' in data and 'option_chains' in data:
        # Get the first expiration's data
        first_exp = list(data['option_chains'].keys())[0]
        option_data = data['option_chains'][first_exp]['calls']
        
        engineered_features = engineer.engineer_all_features(
            data['historical_data'], 
            option_data,
            data.get('sentiment_data', {})
        )
        
        print(f"Engineered features shape: {engineered_features.shape}")
        print(f"Feature columns: {list(engineered_features.columns)}")
