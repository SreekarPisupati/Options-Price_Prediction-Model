# Options Price Prediction Model - Quickstart Guide

## Overview
This project provides a comprehensive machine learning system for predicting options prices with a target of achieving at least 12% performance improvement over the Black-Scholes baseline model.

## Features
- **Data Collection**: Real-time and historical options data via yFinance
- **Feature Engineering**: Greeks, volatility, technical indicators, sentiment analysis
- **ML Models**: TensorFlow, XGBoost, LightGBM, Random Forest, Ensemble
- **Black-Scholes Baseline**: Traditional options pricing for comparison
- **Evaluation Framework**: Performance metrics and statistical significance testing
- **Backtesting**: Trading strategy simulation and performance analysis

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Windows/Mac/Linux
- At least 8GB RAM (16GB recommended)
- 2GB free disk space

### Required Software
- Git
- Python package manager (pip)
- Internet connection for data fetching

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/SreekarPisupati/Options-Price_Prediction-Model.git
cd Options-Price_Prediction-Model
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; import xgboost as xgb; import yfinance as yf; print('All packages installed successfully!')"
```

## Project Structure
```
Options-Price_Prediction-Model/
├── src/                          # Source code
│   ├── main.py                   # Main orchestration script
│   ├── data_collector.py         # Data collection from yFinance
│   ├── feature_engineering.py   # Feature engineering pipeline
│   ├── ml_models.py             # Machine learning models
│   ├── black_scholes.py         # Black-Scholes baseline model
│   ├── evaluation.py            # Evaluation and backtesting
│   ├── real_time_engine.py      # Real-time prediction engine
│   └── config.py               # Configuration settings
├── data/                        # Data storage
│   ├── raw/                     # Raw data files
│   ├── processed/              # Processed features
│   └── predictions/            # Prediction outputs
├── models/                     # Trained model storage
├── logs/                       # Application logs
├── docs/                       # Documentation and reports
└── requirements.txt            # Python dependencies
```

## Running the Project

### Quick Start (Recommended)
Run the complete pipeline with default settings:
```bash
cd src
python main.py --mode full --symbols AAPL MSFT GOOGL
```

### Step-by-Step Execution

#### 1. Data Collection
Collect options and historical data for specific symbols:
```bash
python main.py --mode collect --symbols AAPL MSFT GOOGL TSLA NVDA
```
**What this does:**
- Downloads historical price data (1 year)
- Fetches current options chains
- Collects market sentiment data
- Saves data to `data/raw/` directory

#### 2. Feature Engineering
Process raw data into ML-ready features:
```bash
python main.py --mode features --symbols AAPL MSFT GOOGL
```
**What this does:**
- Calculates Greeks (Delta, Gamma, Theta, Vega, Rho)
- Computes volatility metrics (implied, realized, volatility smile)
- Generates technical indicators (RSI, MACD, Bollinger Bands)
- Creates time-based and sentiment features
- Saves engineered features to `data/processed/`

#### 3. Model Training
Train multiple ML models on the processed data:
```bash
python main.py --mode train --symbols AAPL MSFT GOOGL
```
**What this does:**
- Trains Random Forest, XGBoost, LightGBM models
- Trains TensorFlow/Keras neural networks
- Creates ensemble models
- Performs hyperparameter tuning
- Saves trained models to `models/` directory
- Generates training performance logs

#### 4. Model Evaluation
Evaluate trained models against Black-Scholes baseline:
```bash
python main.py --mode evaluate --symbols AAPL MSFT GOOGL
```
**What this does:**
- Loads pre-trained models
- Compares ML predictions vs Black-Scholes baseline
- Calculates performance improvement metrics
- Performs statistical significance testing
- Generates comprehensive performance report

#### 5. Backtesting
Run trading strategy simulation:
```bash
python main.py --mode backtest --symbols AAPL MSFT GOOGL
```
**What this does:**
- Simulates trading strategies based on predictions
- Calculates P&L, win rates, Sharpe ratios
- Performs risk-adjusted performance analysis
- Generates backtesting reports

#### 6. Real-time Prediction
Start real-time prediction engine:
```bash
python main.py --mode realtime --symbols AAPL --interval 60
```
**What this does:**
- Fetches live market data every 60 seconds
- Makes real-time options price predictions
- Compares with current market prices
- Logs prediction accuracy in real-time

### Advanced Usage

#### Custom Configuration
Edit `src/config.py` to customize:
- Data collection parameters
- Model hyperparameters
- Feature engineering settings
- Risk management rules

#### Multiple Symbol Analysis
```bash
python main.py --mode full --symbols AAPL MSFT GOOGL TSLA NVDA AMZN META --data-period 2y
```

#### Specific Model Training
```bash
python main.py --mode train --symbols AAPL --models xgboost neural_network
```

#### Extended Backtesting
```bash
python main.py --mode backtest --symbols AAPL MSFT --start-date 2023-01-01 --end-date 2024-01-01
```

## Understanding the Output

### Performance Metrics
- **MSE/RMSE**: Mean Squared/Root Mean Squared Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of determination (explained variance)
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct price direction predictions

### Key Files Generated
1. **`models/`**: Trained model files (.pkl for sklearn, .h5 for TensorFlow)
2. **`data/processed/`**: Feature-engineered datasets
3. **`logs/`**: Detailed execution logs
4. **`docs/performance_report_*.txt`**: Comprehensive performance reports

### Success Indicators
✅ **Target Achievement**: MSE improvement ≥ 12% vs Black-Scholes  
✅ **Model Performance**: R² > 0.7 for best models  
✅ **Statistical Significance**: p-value < 0.05 in comparison tests  
✅ **Backtesting**: Positive Sharpe ratio and win rate > 50%

## Troubleshooting

### Common Issues

#### 1. Data Collection Errors
```
Error: No data found for symbol AAPL
```
**Solution**: Check internet connection and symbol validity

#### 2. Memory Errors
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce batch size or use fewer symbols

#### 3. Model Training Failures
```
ValueError: Input contains NaN values
```
**Solution**: Data will be automatically cleaned, but check data quality

#### 4. TensorFlow Warnings
```
Could not locate function 'mse'
```
**Solution**: This is handled automatically with `compile=False` in model loading

### Performance Optimization
- **Use fewer symbols** for faster execution
- **Reduce data period** for quicker training
- **Enable GPU** for TensorFlow models (if available)
- **Increase batch size** on high-memory systems

## Expected Runtime
- **Data Collection**: 2-5 minutes per symbol
- **Feature Engineering**: 1-3 minutes per symbol  
- **Model Training**: 10-30 minutes (depends on data size)
- **Evaluation**: 2-5 minutes
- **Full Pipeline**: 20-45 minutes for 3 symbols

## Next Steps
1. **Analyze Results**: Review performance reports in `docs/`
2. **Optimize Models**: Tune hyperparameters based on results
3. **Expand Symbols**: Test on more diverse option chains
4. **Live Trading**: Use real-time engine for actual trading decisions
5. **Custom Features**: Add domain-specific features for your use case

## Support
- Check logs in `logs/` for detailed error information
- Review documentation in `docs/` for methodology details
- Ensure all dependencies are properly installed
- Verify data quality and market hours for data collection

## Performance Goals
🎯 **Primary Goal**: Achieve ≥12% MSE improvement over Black-Scholes baseline  
📊 **Secondary Goals**: R² > 0.7, directional accuracy > 60%, positive backtesting returns
