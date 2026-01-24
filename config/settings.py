"""
Configuration settings for LSTM Trading Platform
"""

# Company configurations
COMPANIES = {
    "AAPL": "Apple",
    # "JPM": "JPMorgan Chase & Co",
    # "PEP": "Pepsi",
    # "TM": "Toyota",
    # "AMZN": "Amazon"
}

COMPANY_ALIASES = {
    "APPLE": ["AAPL"],
    # "JPMORGAN": ["JPM", "JPM.N"],
    # "PEPSI": ["PEP", "PEPSICO"],
    # "TOYOTA": ["TM", "TYO"],
    # "AMAZON": ["AMZN"]
}

# File paths
DATA_FILE = "stock_analysis_complete.xlsx"
COMPANY_TICKERS_JSON = "company_tickers.json"

# Data collection settings
FORCE_REFETCH = False  # Set to True to force re-fetching all data
FORCE_REGENERATE_ALPHAS = True

# Date ranges (if not loading from cache)
START_DATE = "2023-09-01"
END_DATE = "2025-09-01"

# API settings
NEWS_BATCH_SIZE = 1000
NEWS_SLEEP_TIME = 12  # seconds between batches
RELATED_COMPANIES_SLEEP_TIME = 15  # seconds between related company fetches

# Validation thresholds
MIN_MARKET_CAP = 5e9  # $5 billion
REQUIRED_EXCHANGES = ['NYSE', 'NASDAQ', 'NYQ', 'NMS']
MIN_TRADING_VOLUME = 100000

# Model settings
WINDOW_SIZE = 30
NUM_EPOCHS = 150
LEARNING_RATE = 3e-4
BATCH_SIZE_MIN = 16
BATCH_SIZE_MAX = 64

# Model architecture
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
NUM_HEADS = 4

# Feature selection
USE_FEATURE_SELECTION = True
TOP_K_FEATURES = 30
MIN_CORRELATION = 0.01

# Ensemble settings
ENSEMBLE_SIZE = 2  # Number of models in ensemble
ENSEMBLE_MC_SAMPLES = 10  # Monte Carlo samples for uncertainty

# Training settings
PATIENCE = 30  # Early stopping patience
GRADIENT_CLIP_NORM = 1.0

# Loss function weights
LOSS_ALPHA = 0.7  # MSE weight
LOSS_BETA = 0.3   # Direction weight
LOSS_GAMMA = 0.1  # Large move weight