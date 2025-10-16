import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import re

# ============================================================================
# LSTM Model Architecture
# ============================================================================

class DualStreamLSTM(nn.Module):
    """
    Dual-Stream LSTM for stock price prediction
    - Stream 1: Processes alpha signals
    - Stream 2: Processes price data with temporal features
    """
    def __init__(self, num_alphas=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(DualStreamLSTM, self).__init__()

        # Alpha stream LSTM
        self.alpha_lstm = nn.LSTM(
            input_size=num_alphas,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Price stream LSTM (close + 3 temporal features)
        self.price_lstm = nn.LSTM(
            input_size=4,  # close, day_of_week, day_of_month, month
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fusion layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 1)

    def forward(self, alphas, prices_temporal):
        """
        Args:
            alphas: [batch, window_size, num_alphas]
            prices_temporal: [batch, window_size, 4]
        Returns:
            predictions: [batch, 1]
        """
        # Process alpha stream
        _, (alpha_hidden, _) = self.alpha_lstm(alphas)
        alpha_features = alpha_hidden[-1]  # [batch, hidden_size]

        # Process price stream
        _, (price_hidden, _) = self.price_lstm(prices_temporal)
        price_features = price_hidden[-1]  # [batch, hidden_size]

        # Combine both streams
        combined = torch.cat([alpha_features, price_features], dim=1)  # [batch, hidden_size*2]

        # Dense layers with batch normalization
        out = self.fc1(combined)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)  # [batch, 1]

        return out


class SimpleLSTM(nn.Module):
    """
    Simple stacked LSTM that processes all features together
    """
    def __init__(self, num_features, hidden_size=128, num_layers=2, dropout=0.2):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Args:
            x: [batch, window_size, num_features]
        Returns:
            predictions: [batch, 1]
        """
        lstm_out, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]  # [batch, hidden_size]

        out = self.fc1(last_hidden)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out


# ============================================================================
# Dataset Class
# ============================================================================

class StockDataset(Dataset):
    """
    Dataset for stock prediction with sliding windows
    """
    def __init__(self, data, window_size=5, dual_stream=True):
        """
        Args:
            data: DataFrame with all features including alphas and target
            window_size: Number of days to look back
            dual_stream: Whether to separate alphas and prices for dual-stream model
        """
        self.window_size = window_size
        self.dual_stream = dual_stream

        # Separate features
        alpha_cols = [col for col in data.columns if col.startswith('alpha_')]
        price_col = 'close'
        temporal_cols = ['day_of_week', 'day_of_month', 'month']
        target_col = 'target'

        self.alphas = data[alpha_cols].values
        self.prices = data[[price_col]].values
        self.temporal = data[temporal_cols].values
        self.targets = data[target_col].values

        # Create sliding windows
        self.samples = []
        for i in range(len(data) - window_size):
            self.samples.append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        end_idx = start_idx + self.window_size

        if self.dual_stream:
            # Separate alpha and price streams
            alphas_window = self.alphas[start_idx:end_idx]

            # Combine price with temporal features
            prices_temporal_window = np.concatenate([
                self.prices[start_idx:end_idx],
                self.temporal[start_idx:end_idx]
            ], axis=1)

            target = self.targets[end_idx]

            return (
                torch.FloatTensor(alphas_window),
                torch.FloatTensor(prices_temporal_window),
                torch.FloatTensor([target])
            )
        else:
            # All features together
            all_features = np.concatenate([
                self.alphas[start_idx:end_idx],
                self.prices[start_idx:end_idx],
                self.temporal[start_idx:end_idx]
            ], axis=1)

            target = self.targets[end_idx]

            return (
                torch.FloatTensor(all_features),
                torch.FloatTensor([target])
            )


# ============================================================================
# Alpha Formula Parser and Computer
# ============================================================================

class AlphaComputer:
    """
    Parses and computes alpha formulas generated by LLM
    """
    def __init__(self, df):
        self.df = df.copy()
        self.available_columns = set(df.columns)

    def parse_alpha_formula(self, formula_str):
        """
        Extract formula from text
        """
        # Remove alpha label (e.g., "α1 = " or "Alpha 1:")
        formula = re.sub(r'^[αA]lpha\s*\d+[\s:=]+', '', formula_str, flags=re.IGNORECASE)
        return formula.strip()

    def check_formula_columns(self, formula):
        """
        Check if all columns referenced in formula exist in DataFrame
        Returns: (is_valid, missing_columns)
        """
        # Look for missing sentiment columns
        sentiment_pattern = r'([A-Z]+)_Sentiment'
        sentiment_matches = re.findall(sentiment_pattern, formula)

        missing_cols = []
        for ticker in sentiment_matches:
            col_name = f'{ticker}_Sentiment'
            if col_name not in self.available_columns:
                missing_cols.append(col_name)

        # Check for column names that look like DataFrame columns
        potential_cols = re.findall(r'\b[A-Z][A-Za-z0-9_]*\b', formula)
        for col in potential_cols:
            if col in ['Return', 'SMA', 'EMA', 'MACD', 'BB', 'RSI']:
                # These are prefixes, check if full column exists
                continue
            if col + '_' in formula and col not in self.available_columns:
                # Might be a column prefix
                continue

        return len(missing_cols) == 0, missing_cols

    def compute_alpha(self, formula):
        """
        Compute alpha value from formula string
        Uses pandas eval for safe evaluation with proper error handling
        """
        try:
            # Replace common mathematical symbols
            formula = formula.replace('×', '*').replace('÷', '/')
            formula = formula.replace('−', '-')  # Unicode minus

            # Check if all required columns exist
            is_valid, missing_cols = self.check_formula_columns(formula)
            if not is_valid:
                print(f"    Missing columns: {', '.join(missing_cols)}")
                return None

            # Compute using pandas eval
            result = self.df.eval(formula, engine='python')

            # Check if result is valid
            if result is None:
                return None

            # If result is a DataFrame, take first column
            if isinstance(result, pd.DataFrame):
                if result.shape[1] > 0:
                    result = result.iloc[:, 0]
                else:
                    return None

            # Convert to Series if needed
            if isinstance(result, (int, float)):
                result = pd.Series([result] * len(self.df), index=self.df.index)
            elif not isinstance(result, pd.Series):
                try:
                    result = pd.Series(result, index=self.df.index)
                except:
                    return None

            # Check for NaN/Inf values
            if result.isna().all():
                return None

            return result

        except NameError as e:
            # Column doesn't exist
            missing_col = str(e).split("'")[1] if "'" in str(e) else "unknown"
            return None
        except SyntaxError as e:
            return None
        except Exception as e:
            return None

    def add_alphas_from_text(self, alpha_text, max_alphas=5):
        """
        Parse LLM output text and add computed alphas to dataframe
        """
        lines = alpha_text.split('\n')

        alphas_added = 0
        attempted = 0

        # Show available sentiment columns
        sentiment_cols = [col for col in self.available_columns if 'Sentiment' in col]
        if sentiment_cols:
            print(f"  Available sentiment columns: {', '.join(sentiment_cols)}")

        for line in lines:
            if alphas_added >= max_alphas:
                break

            # Look for lines that contain formulas (with = sign)
            if '=' in line and any(char.isalpha() for char in line):
                attempted += 1

                try:
                    formula = self.parse_alpha_formula(line)

                    # Skip empty formulas
                    if not formula or len(formula) < 3:
                        continue

                    # Compute alpha
                    alpha_values = self.compute_alpha(formula)

                    if alpha_values is not None:
                        alphas_added += 1
                        col_name = f'alpha_{alphas_added}'
                        self.df[col_name] = alpha_values
                        print(f"  ✓ Added {col_name}: {formula[:80]}...")
                    else:
                        print(f"  ✗ Could not compute: {line[:80]}...")

                except Exception as e:
                    print(f"  ✗ Error: {line[:80]}... ({e})")
                    continue

        print(f"  Summary: Added {alphas_added}/{attempted} alphas")

        if alphas_added == 0:
            print(f"  ⚠️  WARNING: No alphas computed! Check if LLM used correct column names.")

        return self.df, alphas_added

# ============================================================================
# Data Preparation Functions
# ============================================================================

def add_temporal_features(df):
    """
    Add temporal features: day of week, day of month, month
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    df['day_of_week'] = df['date'].dt.dayofweek / 6.0  # Normalize 0-1
    df['day_of_month'] = df['date'].dt.day / 31.0      # Normalize 0-1
    df['month'] = df['date'].dt.month / 12.0           # Normalize 0-1

    return df

def prepare_data_for_training(df, ticker, alpha_text, window_size=5):
    """
    Prepare data for LSTM training

    Args:
        df: DataFrame with stock data and sentiment
        ticker: Stock ticker symbol
        alpha_text: LLM-generated alpha formulas (text)
        window_size: Sliding window size

    Returns:
        train_loader, val_loader, test_loader, scalers, num_alphas
    """
    # Add temporal features
    df = add_temporal_features(df)

    # Compute alphas from LLM text
    print(f"\nComputing alphas for {ticker}...")
    alpha_computer = AlphaComputer(df)
    df, num_alphas = alpha_computer.add_alphas_from_text(alpha_text, max_alphas=5)

    if num_alphas == 0:
        print(f"Warning: No alphas computed for {ticker}")
        return None, None, None, None, 0

    # Create target (next day close price)
    df['target'] = df['close'].shift(-1)

    # Drop rows with NaN
    df = df.dropna()

    if len(df) < window_size + 50:
        print(f"Warning: Insufficient data for {ticker} ({len(df)} rows)")
        return None, None, None, None, 0

    # Scale features
    alpha_cols = [col for col in df.columns if col.startswith('alpha_')]

    scaler_alphas = StandardScaler()
    scaler_price = StandardScaler()

    # Convert to proper numpy arrays to avoid dtype issues
    alpha_values = df[alpha_cols].values.astype(float)
    df[alpha_cols] = scaler_alphas.fit_transform(alpha_values)

    # Scale close and target together (they're both prices)
    # Fit scaler on BOTH columns at once
    price_cols = ['close', 'target']
    price_values = df[price_cols].values.astype(float)
    scaled_prices = scaler_price.fit_transform(price_values)

    # Assign scaled values back
    df[price_cols] = scaled_prices

    # Split data: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]

    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets
    train_dataset = StockDataset(train_df, window_size=window_size, dual_stream=True)
    val_dataset = StockDataset(val_df, window_size=window_size, dual_stream=True)
    test_dataset = StockDataset(test_df, window_size=window_size, dual_stream=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    scalers = {
        'alphas': scaler_alphas,
        'price': scaler_price
    }

    return train_loader, val_loader, test_loader, scalers, num_alphas


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in train_loader:
        alphas, prices_temporal, targets = batch
        alphas = alphas.to(device)
        prices_temporal = prices_temporal.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(alphas, prices_temporal)
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            alphas, prices_temporal, targets = batch
            alphas = alphas.to(device)
            prices_temporal = prices_temporal.to(device)
            targets = targets.to(device)

            predictions = model(alphas, prices_temporal)
            loss = criterion(predictions, targets)

            total_loss += loss.item()

    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=5e-5, device='cpu'):
    """
    Train the LSTM model
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    print(f"\nTraining on {device}...")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    return model, train_losses, val_losses

def evaluate_model(model, test_loader, scaler_price, device):
    """
    Evaluate model on test set and calculate metrics
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in test_loader:
            alphas, prices_temporal, targets = batch
            alphas = alphas.to(device)
            prices_temporal = prices_temporal.to(device)

            preds = model(alphas, prices_temporal)

            predictions.extend(preds.cpu().numpy())
            actuals.extend(targets.numpy())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Inverse transform to get actual prices
    predictions_unscaled = scaler_price.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_unscaled = scaler_price.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = np.mean((predictions_unscaled - actuals_unscaled) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_unscaled - actuals_unscaled))
    mape = np.mean(np.abs((actuals_unscaled - predictions_unscaled) / actuals_unscaled)) * 100

    # Directional accuracy
    actual_direction = np.sign(np.diff(actuals_unscaled))
    pred_direction = np.sign(np.diff(predictions_unscaled))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional Accuracy': directional_accuracy
    }

    return predictions_unscaled, actuals_unscaled, metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_predictions(actuals, predictions, ticker, save_path=None):
    """
    Plot actual vs predicted prices
    """
    plt.figure(figsize=(14, 6))

    plt.plot(actuals, label='Actual Price', linewidth=2, alpha=0.7)
    plt.plot(predictions, label='Predicted Price', linewidth=2, alpha=0.7)

    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.title(f'{ticker} Stock Price Prediction', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_training_history(train_losses, val_losses, ticker, save_path=None):
    """
    Plot training and validation loss over epochs
    """
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'{ticker} Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_stock_predictor(ticker, company_name, comprehensive_df, alpha_text,
                         window_size=5, num_epochs=50, device='cpu'):
    """
    Complete pipeline to train stock price predictor

    Args:
        ticker: Stock ticker symbol
        company_name: Company name
        comprehensive_df: DataFrame with stock data, technical indicators, sentiment
        alpha_text: LLM-generated alpha formulas
        window_size: Sliding window size (default: 5 days)
        num_epochs: Number of training epochs
        device: 'cpu' or 'cuda'

    Returns:
        model, metrics, scalers
    """
    print("="*80)
    print(f"TRAINING LSTM PREDICTOR FOR {company_name} ({ticker})")
    print("="*80)

    # Prepare data
    train_loader, val_loader, test_loader, scalers, num_alphas = prepare_data_for_training(
        comprehensive_df, ticker, alpha_text, window_size
    )

    if train_loader is None:
        print(f"Failed to prepare data for {ticker}")
        return None, None, None

    # Initialize model
    model = DualStreamLSTM(
        num_alphas=num_alphas,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    print(f"\nModel Architecture:")
    print(f"  - Alpha stream: {num_alphas} inputs")
    print(f"  - Price stream: 4 inputs (close + 3 temporal)")
    print(f"  - Hidden size: 64")
    print(f"  - Num layers: 2")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=5e-5,
        device=device
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions, actuals, metrics = evaluate_model(
        model, test_loader, scalers['price'], device
    )

    # Print metrics
    print("\n" + "="*80)
    print(f"TEST SET RESULTS FOR {ticker}")
    print("="*80)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("="*80)

    # Plot results
    plot_training_history(train_losses, val_losses, ticker,
                         save_path=f'{ticker}_training_history.png')
    plot_predictions(actuals, predictions, ticker,
                    save_path=f'{ticker}_predictions.png')

    return model, metrics, scalers


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Load data and train
    # This assumes you have comprehensive_df and alpha_text from the previous script

    # Example alpha text (normally from LLM)
    example_alpha_text = """
    α1 = Return_5D + 0.5 × (AAPL_Sentiment - MSFT_Sentiment)
    α2 = (close - SMA_20) / SMA_20
    α3 = RSI / 100 - 0.5
    α4 = (MACD - MACD_Signal) / close
    α5 = Volatility_20D × (AAPL_Sentiment - 0.5)
    """

    # Example dataframe (you would load this from your data)
    # comprehensive_df = pd.read_csv('stock_data.csv')

    # Train the model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model, metrics, scalers = train_stock_predictor(
    #     ticker='AAPL',
    #     company_name='Apple Inc.',
    #     comprehensive_df=comprehensive_df,
    #     alpha_text=example_alpha_text,
    #     window_size=5,
    #     num_epochs=50,
    #     device=device
    # )

    print("LSTM Stock Predictor Module Ready!")
    print("Use train_stock_predictor() to train models for your stocks.")