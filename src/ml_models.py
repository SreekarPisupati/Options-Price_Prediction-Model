"""
Machine Learning Models for Options Price Prediction

This module implements various ML models including:
- Neural Networks (TensorFlow/Keras)
- Random Forests (Scikit-learn)
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Regression
- Ensemble methods
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class OptionsMLModels:
    """
    Comprehensive ML models for options price prediction
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ML models
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.model_performances = {}
        self.logger = self._setup_logging()
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger
    
    def build_neural_network(self, input_shape: int, architecture: str = 'deep') -> keras.Model:
        """
        Build neural network architecture
        
        Args:
            input_shape: Number of input features
            architecture: Type of architecture ('simple', 'deep', 'wide_deep')
            
        Returns:
            Compiled Keras model
        """
        try:
            if architecture == 'simple':
                model = keras.Sequential([
                    layers.Dense(64, activation='relu', input_shape=(input_shape,)),
                    layers.Dropout(0.2),
                    layers.Dense(32, activation='relu'),
                    layers.Dropout(0.1),
                    layers.Dense(1, activation='linear')
                ])
                
            elif architecture == 'deep':
                model = keras.Sequential([
                    layers.Dense(256, activation='relu', input_shape=(input_shape,)),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(128, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(0.1),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1, activation='linear')
                ])
                
            elif architecture == 'wide_deep':
                # Wide component
                wide_input = keras.Input(shape=(input_shape,), name='wide_input')
                wide = layers.Dense(1, activation='linear', name='wide')(wide_input)
                
                # Deep component
                deep = layers.Dense(256, activation='relu')(wide_input)
                deep = layers.BatchNormalization()(deep)
                deep = layers.Dropout(0.3)(deep)
                deep = layers.Dense(128, activation='relu')(deep)
                deep = layers.BatchNormalization()(deep)
                deep = layers.Dropout(0.2)(deep)
                deep = layers.Dense(64, activation='relu')(deep)
                deep = layers.Dropout(0.1)(deep)
                deep = layers.Dense(32, activation='relu')(deep)
                deep = layers.Dense(1, activation='linear', name='deep')(deep)
                
                # Combine wide and deep
                output = layers.Add()([wide, deep])
                model = keras.Model(inputs=wide_input, outputs=output)
            
            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mse']
            )
            
            self.logger.info(f"Built {architecture} neural network with {input_shape} inputs")
            return model
            
        except Exception as e:
            self.logger.error(f"Error building neural network: {str(e)}")
            raise e
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           architecture: str = 'deep', epochs: int = 100) -> keras.Model:
        """
        Train neural network model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            architecture: Network architecture
            epochs: Number of training epochs
            
        Returns:
            Trained Keras model
        """
        try:
            model = self.build_neural_network(X_train.shape[1], architecture)
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            )
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=64,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.models[f'neural_network_{architecture}'] = model
            
            # Evaluate performance
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            self.model_performances[f'neural_network_{architecture}'] = {
                'mse': mse, 'mae': mae, 'r2': r2
            }
            
            self.logger.info(f"Neural network ({architecture}) trained - MSE: {mse:.6f}, R²: {r2:.4f}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error training neural network: {str(e)}")
            raise e
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> RandomForestRegressor:
        """
        Train Random Forest model with hyperparameter tuning
        """
        try:
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 0.33]
            }
            
            # Initialize model
            rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            
            # Grid search with time series split
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                rf, param_grid, cv=tscv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=0
            )
            
            # Fit model
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate
            y_pred = best_model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            self.models['random_forest'] = best_model
            self.model_performances['random_forest'] = {
                'mse': mse, 'mae': mae, 'r2': r2,
                'best_params': grid_search.best_params_
            }
            
            self.logger.info(f"Random Forest trained - MSE: {mse:.6f}, R²: {r2:.4f}")
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest: {str(e)}")
            raise e
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBRegressor:
        """
        Train XGBoost model
        """
        try:
            # Parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # Initialize model
            xgb_model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='rmse'
            )
            
            # Grid search
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=tscv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=0
            )
            
            # Fit model
            grid_search.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            best_model = grid_search.best_estimator_
            
            # Evaluate
            y_pred = best_model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            self.models['xgboost'] = best_model
            self.model_performances['xgboost'] = {
                'mse': mse, 'mae': mae, 'r2': r2,
                'best_params': grid_search.best_params_
            }
            
            self.logger.info(f"XGBoost trained - MSE: {mse:.6f}, R²: {r2:.4f}")
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost: {str(e)}")
            # Fallback to simple XGBoost if grid search fails
            simple_model = xgb.XGBRegressor(random_state=self.random_state)
            simple_model.fit(X_train, y_train)
            self.models['xgboost'] = simple_model
            return simple_model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> lgb.LGBMRegressor:
        """
        Train LightGBM model
        """
        try:
            # Best parameters for options prediction
            best_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }
            
            # Initialize and train model
            lgb_model = lgb.LGBMRegressor(**best_params, n_estimators=500)
            
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Evaluate
            y_pred = lgb_model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            self.models['lightgbm'] = lgb_model
            self.model_performances['lightgbm'] = {
                'mse': mse, 'mae': mae, 'r2': r2
            }
            
            self.logger.info(f"LightGBM trained - MSE: {mse:.6f}, R²: {r2:.4f}")
            return lgb_model
            
        except Exception as e:
            self.logger.error(f"Error training LightGBM: {str(e)}")
            # Fallback to simple LightGBM
            simple_model = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
            simple_model.fit(X_train, y_train)
            self.models['lightgbm'] = simple_model
            return simple_model
    
    def train_support_vector_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                      X_val: np.ndarray, y_val: np.ndarray) -> SVR:
        """
        Train Support Vector Regression model
        """
        try:
            # Parameter grid for SVR
            param_grid = {
                'kernel': ['rbf', 'polynomial'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'epsilon': [0.01, 0.1, 0.2]
            }
            
            svr = SVR()
            
            # Grid search
            tscv = TimeSeriesSplit(n_splits=3)
            grid_search = GridSearchCV(
                svr, param_grid, cv=tscv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=0
            )
            
            # Fit model
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate
            y_pred = best_model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            self.models['svr'] = best_model
            self.model_performances['svr'] = {
                'mse': mse, 'mae': mae, 'r2': r2,
                'best_params': grid_search.best_params_
            }
            
            self.logger.info(f"SVR trained - MSE: {mse:.6f}, R²: {r2:.4f}")
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error training SVR: {str(e)}")
            # Fallback to simple SVR
            simple_model = SVR(kernel='rbf', C=1.0)
            simple_model.fit(X_train, y_train)
            self.models['svr'] = simple_model
            return simple_model
    
    def train_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train ensemble model combining multiple algorithms
        """
        try:
            self.logger.info("Training ensemble model...")
            
            # Train individual models if not already trained
            if 'random_forest' not in self.models:
                self.train_random_forest(X_train, y_train, X_val, y_val)
            
            if 'xgboost' not in self.models:
                self.train_xgboost(X_train, y_train, X_val, y_val)
            
            if 'lightgbm' not in self.models:
                self.train_lightgbm(X_train, y_train, X_val, y_val)
            
            # Get predictions from individual models
            rf_pred = self.models['random_forest'].predict(X_val)
            xgb_pred = self.models['xgboost'].predict(X_val)
            lgb_pred = self.models['lightgbm'].predict(X_val)
            
            # Simple averaging ensemble
            ensemble_pred = (rf_pred + xgb_pred + lgb_pred) / 3
            
            # Weighted ensemble based on individual model performance
            rf_r2 = self.model_performances['random_forest']['r2']
            xgb_r2 = self.model_performances['xgboost']['r2']
            lgb_r2 = self.model_performances['lightgbm']['r2']
            
            total_r2 = rf_r2 + xgb_r2 + lgb_r2
            rf_weight = rf_r2 / total_r2
            xgb_weight = xgb_r2 / total_r2
            lgb_weight = lgb_r2 / total_r2
            
            weighted_ensemble_pred = (rf_weight * rf_pred + 
                                    xgb_weight * xgb_pred + 
                                    lgb_weight * lgb_pred)
            
            # Evaluate ensemble performance
            simple_mse = mean_squared_error(y_val, ensemble_pred)
            simple_r2 = r2_score(y_val, ensemble_pred)
            
            weighted_mse = mean_squared_error(y_val, weighted_ensemble_pred)
            weighted_r2 = r2_score(y_val, weighted_ensemble_pred)
            
            # Choose better ensemble
            if weighted_mse < simple_mse:
                best_ensemble = 'weighted'
                best_mse = weighted_mse
                best_r2 = weighted_r2
                weights = {'rf': rf_weight, 'xgb': xgb_weight, 'lgb': lgb_weight}
            else:
                best_ensemble = 'simple'
                best_mse = simple_mse
                best_r2 = simple_r2
                weights = {'rf': 1/3, 'xgb': 1/3, 'lgb': 1/3}
            
            ensemble_info = {
                'type': best_ensemble,
                'weights': weights,
                'models': ['random_forest', 'xgboost', 'lightgbm']
            }
            
            self.models['ensemble'] = ensemble_info
            self.model_performances['ensemble'] = {
                'mse': best_mse, 'mae': mean_absolute_error(y_val, 
                weighted_ensemble_pred if best_ensemble == 'weighted' else ensemble_pred),
                'r2': best_r2
            }
            
            self.logger.info(f"Ensemble model trained - MSE: {best_mse:.6f}, R²: {best_r2:.4f}")
            return ensemble_info
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {str(e)}")
            return {}
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble model
        """
        try:
            if 'ensemble' not in self.models:
                raise ValueError("Ensemble model not trained")
            
            ensemble_info = self.models['ensemble']
            weights = ensemble_info['weights']
            
            # Get predictions from individual models
            rf_pred = self.models['random_forest'].predict(X)
            xgb_pred = self.models['xgboost'].predict(X)
            lgb_pred = self.models['lightgbm'].predict(X)
            
            # Weighted ensemble prediction
            ensemble_pred = (weights['rf'] * rf_pred + 
                           weights['xgb'] * xgb_pred + 
                           weights['lgb'] * lgb_pred)
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Error making ensemble prediction: {str(e)}")
            return np.array([])
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Train all available models and compare performance
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with all model performances
        """
        try:
            self.logger.info("Starting training of all models...")
            
            # Prepare data
            X_numeric = X.select_dtypes(include=[np.number])
            X_clean = X_numeric.fillna(X_numeric.median())
            
            # Split data temporally (for time series)
            split_idx = int(len(X_clean) * (1 - test_size))
            X_train = X_clean.iloc[:split_idx].values
            X_val = X_clean.iloc[split_idx:].values
            y_train = y.iloc[:split_idx].values
            y_val = y.iloc[split_idx:].values
            
            self.logger.info(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
            
            # Train all models
            try:
                self.train_random_forest(X_train, y_train, X_val, y_val)
            except Exception as e:
                self.logger.warning(f"Random Forest training failed: {str(e)}")
            
            try:
                self.train_xgboost(X_train, y_train, X_val, y_val)
            except Exception as e:
                self.logger.warning(f"XGBoost training failed: {str(e)}")
            
            try:
                self.train_lightgbm(X_train, y_train, X_val, y_val)
            except Exception as e:
                self.logger.warning(f"LightGBM training failed: {str(e)}")
            
            try:
                self.train_neural_network(X_train, y_train, X_val, y_val, 'deep')
            except Exception as e:
                self.logger.warning(f"Neural Network training failed: {str(e)}")
            
            # Train ensemble if we have multiple models
            if len(self.models) >= 2:
                try:
                    self.train_ensemble_model(X_train, y_train, X_val, y_val)
                except Exception as e:
                    self.logger.warning(f"Ensemble training failed: {str(e)}")
            
            # Find best model
            if self.model_performances:
                best_model_name = min(self.model_performances.keys(), 
                                    key=lambda x: self.model_performances[x]['mse'])
                self.logger.info(f"Best model: {best_model_name} with MSE: {self.model_performances[best_model_name]['mse']:.6f}")
            
            return self.model_performances
            
        except Exception as e:
            self.logger.error(f"Error training all models: {str(e)}")
            return {}
    
    def predict(self, X: pd.DataFrame, model_name: str = 'best') -> np.ndarray:
        """
        Make predictions using specified model
        
        Args:
            X: Feature matrix
            model_name: Name of model to use ('best', 'ensemble', 'xgboost', etc.)
            
        Returns:
            Array of predictions
        """
        try:
            # Determine which model to use
            if model_name == 'best':
                if self.model_performances:
                    model_name = min(self.model_performances.keys(), 
                                   key=lambda x: self.model_performances[x]['mse'])
                else:
                    raise ValueError("No models have been trained")
            
            if model_name == 'ensemble':
                return self.predict_ensemble(X.values)
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Handle neural networks
            if 'neural_network' in model_name:
                predictions = model.predict(X.values)
                return predictions.flatten()
            else:
                return model.predict(X.values)
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return np.array([])
    
    def get_feature_importance(self, model_name: str = 'xgboost', 
                             feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Args:
            model_name: Name of model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                return importance_df
            else:
                self.logger.warning(f"Model {model_name} does not support feature importance")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def save_models(self, save_path: str = "../models/"):
        """
        Save trained models to disk
        
        Args:
            save_path: Path to save models
        """
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            for model_name, model in self.models.items():
                if 'neural_network' in model_name:
                    model.save(f"{save_path}{model_name}.h5")
                elif model_name != 'ensemble':
                    joblib.dump(model, f"{save_path}{model_name}.pkl")
                else:
                    # Save ensemble info
                    joblib.dump(model, f"{save_path}ensemble_info.pkl")
            
            # Save performance metrics
            joblib.dump(self.model_performances, f"{save_path}model_performances.pkl")
            
            self.logger.info(f"Models saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, load_path: str = "../models/"):
        """
        Load models from disk
        
        Args:
            load_path: Path to load models from
        """
        try:
            import os
            import glob
            
            # Load sklearn models
            pkl_files = glob.glob(f"{load_path}*.pkl")
            for pkl_file in pkl_files:
                model_name = os.path.basename(pkl_file).replace('.pkl', '')
                if model_name == 'model_performances':
                    self.model_performances = joblib.load(pkl_file)
                else:
                    self.models[model_name] = joblib.load(pkl_file)
            
            # Load TensorFlow models
            h5_files = glob.glob(f"{load_path}*.h5")
            for h5_file in h5_files:
                model_name = os.path.basename(h5_file).replace('.h5', '')
                self.models[model_name] = keras.models.load_model(h5_file)
            
            self.logger.info(f"Loaded {len(self.models)} models from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all trained models on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with evaluation metrics
        """
        try:
            results = []
            
            for model_name in self.models.keys():
                if model_name == 'ensemble':
                    y_pred = self.predict_ensemble(X_test)
                else:
                    model = self.models[model_name]
                    if 'neural_network' in model_name:
                        y_pred = model.predict(X_test).flatten()
                    else:
                        y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'model': model_name,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                })
            
            results_df = pd.DataFrame(results).sort_values('mse')
            self.logger.info("Model evaluation completed")
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            return pd.DataFrame()

class AdvancedNeuralNetworks:
    """
    Advanced neural network architectures for options prediction
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        tf.random.set_seed(random_state)
    
    def build_lstm_model(self, sequence_length: int, n_features: int) -> keras.Model:
        """
        Build LSTM model for time series prediction
        """
        model = keras.Sequential([
            layers.LSTM(100, return_sequences=True, input_shape=(sequence_length, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_attention_model(self, input_shape: int) -> keras.Model:
        """
        Build attention-based model
        """
        inputs = keras.Input(shape=(input_shape,))
        
        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1
        )(inputs, inputs)
        
        attention = layers.LayerNormalization()(attention + inputs)
        
        # Feed forward
        ff = layers.Dense(256, activation='relu')(attention)
        ff = layers.Dropout(0.2)(ff)
        ff = layers.Dense(128, activation='relu')(ff)
        ff = layers.LayerNormalization()(ff + attention)
        
        # Output
        outputs = layers.Dense(64, activation='relu')(ff)
        outputs = layers.Dense(1, activation='linear')(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model

if __name__ == "__main__":
    # Example usage
    print("ML Models module initialized successfully")
    
    # Create sample data for testing
    np.random.seed(42)
    X_sample = pd.DataFrame(np.random.randn(1000, 20))
    y_sample = pd.Series(np.random.randn(1000) * 10 + 50)
    
    # Initialize models
    ml_models = OptionsMLModels()
    
    # Train models
    try:
        performances = ml_models.train_all_models(X_sample, y_sample)
        print(f"Trained {len(ml_models.models)} models")
        print("Model performances:")
        for model, perf in performances.items():
            print(f"  {model}: R² = {perf['r2']:.4f}")
    except Exception as e:
        print(f"Error in example: {e}")
