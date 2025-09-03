"""
Black-Scholes Model Implementation

This module provides the Black-Scholes baseline model for option pricing,
including Greeks calculations and model extensions.
Used as baseline for performance comparison.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

class BlackScholesModel:
    """
    Implementation of the Black-Scholes option pricing model
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize Black-Scholes model
        
        Args:
            risk_free_rate: Risk-free interest rate
        """
        self.risk_free_rate = risk_free_rate
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger
    
    def calculate_option_price(self, S: float, K: float, T: float, 
                             sigma: float, option_type: str = 'call',
                             dividend_yield: float = 0.0) -> float:
        """
        Calculate Black-Scholes option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            sigma: Volatility
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
            
        Returns:
            Option price
        """
        try:
            # Adjust for dividend yield
            r = self.risk_free_rate
            q = dividend_yield
            
            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            elif option_type.lower() == 'put':
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
            return max(price, 0.01)  # Minimum price of $0.01
            
        except Exception as e:
            self.logger.error(f"Error calculating option price: {str(e)}")
            return 0.01
    
    def calculate_greeks(self, S: float, K: float, T: float, sigma: float,
                        option_type: str = 'call', dividend_yield: float = 0.0) -> Dict[str, float]:
        """
        Calculate Black-Scholes Greeks
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            sigma: Volatility
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
            
        Returns:
            Dictionary with Greeks values
        """
        try:
            r = self.risk_free_rate
            q = dividend_yield
            
            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Calculate Greeks
            greeks = {}
            
            if option_type.lower() == 'call':
                greeks['delta'] = np.exp(-q*T) * norm.cdf(d1)
                greeks['theta'] = ((-S*np.exp(-q*T)*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) -
                                 r*K*np.exp(-r*T)*norm.cdf(d2) +
                                 q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
                greeks['rho'] = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
            
            elif option_type.lower() == 'put':
                greeks['delta'] = -np.exp(-q*T) * norm.cdf(-d1)
                greeks['theta'] = ((-S*np.exp(-q*T)*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) +
                                 r*K*np.exp(-r*T)*norm.cdf(-d2) -
                                 q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
                greeks['rho'] = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
            
            # Greeks that are the same for calls and puts
            greeks['gamma'] = np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            greeks['vega'] = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T) / 100
            
            return greeks
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {str(e)}")
            return {}
    
    def implied_volatility(self, market_price: float, S: float, K: float, T: float,
                          option_type: str = 'call', dividend_yield: float = 0.0,
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price: Current market price of option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
            max_iterations: Maximum iterations for convergence
            tolerance: Tolerance for convergence
            
        Returns:
            Implied volatility
        """
        try:
            # Initial guess
            sigma = 0.25
            
            for i in range(max_iterations):
                # Calculate option price and vega with current sigma
                price = self.calculate_option_price(S, K, T, sigma, option_type, dividend_yield)
                greeks = self.calculate_greeks(S, K, T, sigma, option_type, dividend_yield)
                vega = greeks.get('vega', 0.01) * 100  # Convert to price per 1% vol change
                
                # Newton-Raphson update
                price_diff = price - market_price
                
                if abs(price_diff) < tolerance:
                    return sigma
                
                if vega == 0:
                    break
                    
                sigma = sigma - price_diff / vega
                
                # Keep sigma positive and reasonable
                sigma = max(min(sigma, 3.0), 0.01)
            
            return sigma
            
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {str(e)}")
            return 0.25  # Return default volatility
    
    def batch_option_pricing(self, option_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Black-Scholes prices for a batch of options
        
        Args:
            option_data: DataFrame with option parameters
            
        Returns:
            DataFrame with calculated prices and Greeks
        """
        try:
            df = option_data.copy()
            
            # Required columns check
            required_cols = ['current_price', 'strike', 'time_to_expiry', 'impliedVolatility']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Initialize result columns
            df['bs_price'] = 0.0
            df['bs_delta'] = 0.0
            df['bs_gamma'] = 0.0
            df['bs_theta'] = 0.0
            df['bs_vega'] = 0.0
            df['bs_rho'] = 0.0
            
            # Calculate for each row
            for idx, row in df.iterrows():
                try:
                    S = row['current_price']
                    K = row['strike']
                    T = max(row['time_to_expiry'], 0.001)  # Minimum 1 day
                    sigma = max(row['impliedVolatility'], 0.01)  # Minimum 1% vol
                    option_type = row.get('type', 'call')
                    
                    # Calculate price
                    price = self.calculate_option_price(S, K, T, sigma, option_type)
                    df.loc[idx, 'bs_price'] = price
                    
                    # Calculate Greeks
                    greeks = self.calculate_greeks(S, K, T, sigma, option_type)
                    for greek, value in greeks.items():
                        df.loc[idx, f'bs_{greek}'] = value
                        
                except Exception as row_error:
                    self.logger.warning(f"Error processing row {idx}: {str(row_error)}")
                    continue
            
            # Calculate model performance metrics
            if 'mid_price' in df.columns:
                df['bs_error'] = df['mid_price'] - df['bs_price']
                df['bs_error_pct'] = df['bs_error'] / df['mid_price']
                df['bs_abs_error'] = np.abs(df['bs_error'])
                df['bs_abs_error_pct'] = np.abs(df['bs_error_pct'])
            
            self.logger.info(f"Calculated Black-Scholes prices for {len(df)} options")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in batch option pricing: {str(e)}")
            return option_data
    
    def calculate_portfolio_greeks(self, positions: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks
        
        Args:
            positions: DataFrame with position data including quantities and Greeks
            
        Returns:
            Dictionary with portfolio Greeks
        """
        try:
            # Required columns
            required_cols = ['quantity', 'bs_delta', 'bs_gamma', 'bs_theta', 'bs_vega', 'bs_rho']
            missing_cols = [col for col in required_cols if col not in positions.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns for portfolio Greeks: {missing_cols}")
                return {}
            
            portfolio_greeks = {
                'delta': (positions['quantity'] * positions['bs_delta']).sum(),
                'gamma': (positions['quantity'] * positions['bs_gamma']).sum(),
                'theta': (positions['quantity'] * positions['bs_theta']).sum(),
                'vega': (positions['quantity'] * positions['bs_vega']).sum(),
                'rho': (positions['quantity'] * positions['bs_rho']).sum()
            }
            
            self.logger.info("Successfully calculated portfolio Greeks")
            return portfolio_greeks
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio Greeks: {str(e)}")
            return {}
    
    def stress_test_portfolio(self, positions: pd.DataFrame, 
                            price_shocks: List[float] = None,
                            vol_shocks: List[float] = None) -> pd.DataFrame:
        """
        Perform stress testing on option portfolio
        
        Args:
            positions: DataFrame with position data
            price_shocks: List of price shock percentages
            vol_shocks: List of volatility shock percentages
            
        Returns:
            DataFrame with stress test results
        """
        try:
            if price_shocks is None:
                price_shocks = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
            
            if vol_shocks is None:
                vol_shocks = [-0.5, -0.25, 0, 0.25, 0.5]
            
            stress_results = []
            
            for price_shock in price_shocks:
                for vol_shock in vol_shocks:
                    # Apply shocks
                    shocked_positions = positions.copy()
                    shocked_positions['current_price'] *= (1 + price_shock)
                    shocked_positions['impliedVolatility'] *= (1 + vol_shock)
                    
                    # Recalculate option prices
                    shocked_positions = self.batch_option_pricing(shocked_positions)
                    
                    # Calculate portfolio P&L
                    if 'quantity' in shocked_positions.columns:
                        pnl = (shocked_positions['quantity'] * 
                              (shocked_positions['bs_price'] - positions['bs_price'])).sum()
                    else:
                        pnl = (shocked_positions['bs_price'] - positions['bs_price']).sum()
                    
                    stress_results.append({
                        'price_shock': price_shock,
                        'vol_shock': vol_shock,
                        'portfolio_pnl': pnl
                    })
            
            stress_df = pd.DataFrame(stress_results)
            self.logger.info("Successfully completed stress testing")
            return stress_df
            
        except Exception as e:
            self.logger.error(f"Error in stress testing: {str(e)}")
            return pd.DataFrame()
    
    def monte_carlo_option_pricing(self, S: float, K: float, T: float, 
                                  sigma: float, option_type: str = 'call',
                                  n_simulations: int = 100000,
                                  n_steps: int = 252) -> Dict[str, float]:
        """
        Monte Carlo option pricing simulation
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            sigma: Volatility
            option_type: 'call' or 'put'
            n_simulations: Number of simulation paths
            n_steps: Number of time steps
            
        Returns:
            Dictionary with MC price and confidence intervals
        """
        try:
            dt = T / n_steps
            r = self.risk_free_rate
            
            # Generate random paths
            np.random.seed(42)
            Z = np.random.standard_normal((n_simulations, n_steps))
            
            # Stock price paths
            log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            log_S = np.log(S) + np.cumsum(log_returns, axis=1)
            S_paths = np.exp(log_S)
            
            # Final stock prices
            S_final = S_paths[:, -1]
            
            # Calculate payoffs
            if option_type.lower() == 'call':
                payoffs = np.maximum(S_final - K, 0)
            elif option_type.lower() == 'put':
                payoffs = np.maximum(K - S_final, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
            # Discount payoffs
            discounted_payoffs = payoffs * np.exp(-r * T)
            
            # Calculate statistics
            mc_price = np.mean(discounted_payoffs)
            mc_std = np.std(discounted_payoffs)
            confidence_95 = 1.96 * mc_std / np.sqrt(n_simulations)
            
            results = {
                'mc_price': mc_price,
                'mc_std': mc_std,
                'confidence_interval_95': confidence_95,
                'lower_bound_95': mc_price - confidence_95,
                'upper_bound_95': mc_price + confidence_95
            }
            
            # Compare with analytical Black-Scholes
            bs_price = self.calculate_option_price(S, K, T, sigma, option_type)
            results['bs_analytical'] = bs_price
            results['mc_bs_diff'] = mc_price - bs_price
            results['mc_bs_diff_pct'] = (mc_price - bs_price) / bs_price
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo pricing: {str(e)}")
            return {}
    
    def calculate_implied_volatility(self, market_price: float, S: float, K: float, 
                                   T: float, option_type: str = 'call',
                                   dividend_yield: float = 0.0) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price: Current market price of option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
            
        Returns:
            Implied volatility
        """
        try:
            # Initial guess
            sigma = 0.25
            max_iterations = 100
            tolerance = 1e-6
            
            for i in range(max_iterations):
                # Calculate price and vega
                price = self.calculate_option_price(S, K, T, sigma, option_type, dividend_yield)
                greeks = self.calculate_greeks(S, K, T, sigma, option_type, dividend_yield)
                vega = greeks.get('vega', 0.01) * 100  # Convert to price sensitivity
                
                # Check convergence
                price_diff = price - market_price
                if abs(price_diff) < tolerance:
                    return sigma
                
                if vega == 0:
                    break
                
                # Newton-Raphson update
                sigma = sigma - price_diff / vega
                
                # Keep sigma in reasonable range
                sigma = max(min(sigma, 3.0), 0.01)
            
            return sigma
            
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {str(e)}")
            return 0.25
    
    def volatility_surface_analysis(self, option_data: pd.DataFrame) -> Dict:
        """
        Analyze volatility surface patterns
        
        Args:
            option_data: DataFrame with option data including strikes and expirations
            
        Returns:
            Dictionary with volatility surface analysis
        """
        try:
            df = option_data.copy()
            
            # Calculate implied volatilities if not present
            if 'market_iv' not in df.columns and 'mid_price' in df.columns:
                df['market_iv'] = df.apply(lambda row: self.calculate_implied_volatility(
                    row['mid_price'], row['current_price'], row['strike'],
                    row['time_to_expiry'], row.get('type', 'call')
                ), axis=1)
            
            surface_analysis = {}
            
            # Volatility smile analysis
            if 'market_iv' in df.columns:
                # Group by expiration
                for exp_date in df['expiration'].unique():
                    exp_data = df[df['expiration'] == exp_date].copy()
                    
                    if len(exp_data) > 3:  # Need sufficient data points
                        # Calculate moneyness if not present
                        if 'moneyness' not in exp_data.columns:
                            exp_data['moneyness'] = exp_data['strike'] / exp_data['current_price']
                        
                        # Fit polynomial to volatility smile
                        moneyness = exp_data['moneyness'].values
                        iv = exp_data['market_iv'].values
                        
                        # Remove outliers
                        valid_mask = (iv > 0.05) & (iv < 2.0) & (moneyness > 0.5) & (moneyness < 2.0)
                        if valid_mask.sum() > 3:
                            poly_coeffs = np.polyfit(moneyness[valid_mask], iv[valid_mask], 2)
                            
                            surface_analysis[exp_date] = {
                                'smile_curvature': poly_coeffs[0],  # Second derivative
                                'smile_slope': poly_coeffs[1],      # First derivative
                                'atm_vol': poly_coeffs[2],          # ATM volatility
                                'data_points': valid_mask.sum()
                            }
            
            # Term structure analysis
            if len(surface_analysis) > 1:
                expirations = sorted(surface_analysis.keys())
                atm_vols = [surface_analysis[exp]['atm_vol'] for exp in expirations]
                
                if len(atm_vols) > 1:
                    # Term structure slope
                    term_slope = (atm_vols[-1] - atm_vols[0]) / len(atm_vols)
                    surface_analysis['term_structure_slope'] = term_slope
                    
                    # Volatility term structure inversion
                    inversions = sum(1 for i in range(1, len(atm_vols)) if atm_vols[i] < atm_vols[i-1])
                    surface_analysis['term_structure_inversions'] = inversions
            
            self.logger.info("Successfully analyzed volatility surface")
            return surface_analysis
            
        except Exception as e:
            self.logger.error(f"Error in volatility surface analysis: {str(e)}")
            return {}
    
    def evaluate_model_performance(self, predictions: pd.DataFrame, 
                                 actual_prices: pd.Series) -> Dict[str, float]:
        """
        Evaluate Black-Scholes model performance
        
        Args:
            predictions: DataFrame with BS predictions
            actual_prices: Series with actual market prices
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if 'bs_price' not in predictions.columns:
                raise ValueError("BS predictions not found in DataFrame")
            
            bs_prices = predictions['bs_price']
            
            # Calculate performance metrics
            mse = np.mean((bs_prices - actual_prices) ** 2)
            mae = np.mean(np.abs(bs_prices - actual_prices))
            rmse = np.sqrt(mse)
            
            # Percentage errors
            pct_errors = (bs_prices - actual_prices) / actual_prices
            mape = np.mean(np.abs(pct_errors)) * 100
            
            # R-squared
            ss_res = np.sum((actual_prices - bs_prices) ** 2)
            ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Directional accuracy
            price_changes = np.diff(actual_prices)
            bs_changes = np.diff(bs_prices)
            directional_accuracy = np.mean(np.sign(price_changes) == np.sign(bs_changes))
            
            performance_metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'mean_error': np.mean(bs_prices - actual_prices),
                'std_error': np.std(bs_prices - actual_prices)
            }
            
            self.logger.info(f"Black-Scholes performance - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {str(e)}")
            return {}

class BlackScholesExtensions:
    """
    Extensions to the basic Black-Scholes model
    """
    
    def __init__(self, base_model: BlackScholesModel):
        self.base_model = base_model
        self.logger = base_model.logger
    
    def american_option_binomial(self, S: float, K: float, T: float, sigma: float,
                               option_type: str = 'call', n_steps: int = 100) -> float:
        """
        American option pricing using binomial tree
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            sigma: Volatility
            option_type: 'call' or 'put'
            n_steps: Number of steps in binomial tree
            
        Returns:
            American option price
        """
        try:
            r = self.base_model.risk_free_rate
            dt = T / n_steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)
            
            # Initialize stock price tree
            stock_tree = np.zeros((n_steps + 1, n_steps + 1))
            stock_tree[0, 0] = S
            
            # Fill stock price tree
            for i in range(1, n_steps + 1):
                stock_tree[i, 0] = stock_tree[i-1, 0] * u
                for j in range(1, i + 1):
                    stock_tree[i, j] = stock_tree[i-1, j-1] * d
            
            # Initialize option value tree
            option_tree = np.zeros((n_steps + 1, n_steps + 1))
            
            # Calculate option values at expiration
            for j in range(n_steps + 1):
                if option_type.lower() == 'call':
                    option_tree[n_steps, j] = max(stock_tree[n_steps, j] - K, 0)
                else:
                    option_tree[n_steps, j] = max(K - stock_tree[n_steps, j], 0)
            
            # Backward induction
            for i in range(n_steps - 1, -1, -1):
                for j in range(i + 1):
                    # European value
                    european_value = np.exp(-r * dt) * (p * option_tree[i+1, j] + 
                                                       (1-p) * option_tree[i+1, j+1])
                    
                    # Early exercise value
                    if option_type.lower() == 'call':
                        exercise_value = max(stock_tree[i, j] - K, 0)
                    else:
                        exercise_value = max(K - stock_tree[i, j], 0)
                    
                    # American option value (max of European and exercise)
                    option_tree[i, j] = max(european_value, exercise_value)
            
            return option_tree[0, 0]
            
        except Exception as e:
            self.logger.error(f"Error in American option pricing: {str(e)}")
            return 0.0
    
    def black_scholes_merton(self, S: float, K: float, T: float, sigma: float,
                           dividend_yield: float, option_type: str = 'call') -> float:
        """
        Black-Scholes-Merton model with dividend yield
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration
            sigma: Volatility
            dividend_yield: Continuous dividend yield
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        return self.base_model.calculate_option_price(
            S, K, T, sigma, option_type, dividend_yield
        )

if __name__ == "__main__":
    # Example usage and testing
    print("Black-Scholes module initialized successfully")
    
    # Initialize model
    bs_model = BlackScholesModel(risk_free_rate=0.05)
    
    # Test single option pricing
    S, K, T, sigma = 100, 100, 0.25, 0.2  # ATM call, 3 months
    
    call_price = bs_model.calculate_option_price(S, K, T, sigma, 'call')
    put_price = bs_model.calculate_option_price(S, K, T, sigma, 'put')
    
    print(f"Call price: ${call_price:.2f}")
    print(f"Put price: ${put_price:.2f}")
    
    # Test Greeks calculation
    call_greeks = bs_model.calculate_greeks(S, K, T, sigma, 'call')
    print("Call Greeks:", call_greeks)
    
    # Test Monte Carlo pricing
    mc_results = bs_model.monte_carlo_option_pricing(S, K, T, sigma, 'call')
    print(f"Monte Carlo call price: ${mc_results.get('mc_price', 0):.2f}")
    print(f"95% Confidence interval: [{mc_results.get('lower_bound_95', 0):.2f}, {mc_results.get('upper_bound_95', 0):.2f}]")
