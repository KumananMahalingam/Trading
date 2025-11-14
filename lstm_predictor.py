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

class ImprovedDualStreamLSTM(nn.Module):

    def __init__(self, num_alphas=5, hidden_size=128, num_layers=3, dropout=0.3):
        super(ImprovedDualStreamLSTM, self).__init__()

        self.hidden_size = hidden_size

        # Alpha stream with bidirectional LSTM
        self.alpha_lstm = nn.LSTM(
            input_size=num_alphas,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Price stream with bidirectional LSTM
        self.price_lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism for alpha stream
        self.alpha_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Attention mechanism for price stream
        self.price_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc1 = nn.Linear(hidden_size * 4, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def attention_net(self, lstm_output, attention_layer):
        """
        Apply attention mechanism to LSTM output
        """
        attention_weights = attention_layer(lstm_output)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)  # [batch, hidden*2]
        return context

    def forward(self, alphas, prices_temporal):
        """
        Args:
            alphas: [batch, window_size, num_alphas]
            prices_temporal: [batch, window_size, 4]
        Returns:
            predictions: [batch, 1]
        """
        # Process alpha stream
        alpha_out, _ = self.alpha_lstm(alphas)
        alpha_features = self.attention_net(alpha_out, self.alpha_attention)

        # Process price stream
        price_out, _ = self.price_lstm(prices_temporal)
        price_features = self.attention_net(price_out, self.price_attention)

        # Combine both streams
        combined = torch.cat([alpha_features, price_features], dim=1)

        # Dense layers
        out = self.fc1(combined)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout3(out)

        out = self.fc4(out)

        return out

class StockDataset(Dataset):
    """
    Dataset for stock prediction with sliding windows
    """
    def __init__(self, data, window_size=20, dual_stream=True):
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
        # Remove alpha label
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

        return len(missing_cols) == 0, missing_cols

    def compute_alpha(self, formula):
        """
        Compute alpha value from formula string
        """
        try:
            # Replace common mathematical symbols
            formula = formula.replace('×', '*').replace('÷', '/')
            formula = formula.replace('−', '-')

            # Check if all required columns exist
            is_valid, missing_cols = self.check_formula_columns(formula)
            if not is_valid:
                print(f"    Missing columns: {', '.join(missing_cols)}")
                return None

            # Compute using pandas eval
            result = self.df.eval(formula, engine='python')

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
            print(f"  Available sentiment columns: {', '.join(sentiment_cols[:10])}...")

        for line in lines:
            if alphas_added >= max_alphas:
                break

            # Look for lines that contain formulas
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
                    print(f"  ✗ Error: {line[:80]}...")
                    continue

        print(f"  Summary: Added {alphas_added}/{attempted} alphas")

        if alphas_added == 0:
            print(f"  ⚠️  WARNING: No alphas computed! Check if LLM used correct column names.")

        return self.df, alphas_added

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

def prepare_data_for_training(df, ticker, alpha_text, window_size=20):
    """
    IMPROVED: Prepare data with no data leakage and percentage change target
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

    df['target'] = df['close'].pct_change(1).shift(-1)

    df = df.dropna()

    if len(df) < window_size + 100:
        print(f"Warning: Insufficient data for {ticker} ({len(df)} rows)")
        return None, None, None, None, 0

    # Split data: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()

    alpha_cols = [col for col in df.columns if col.startswith('alpha_')]

    scaler_alphas = StandardScaler()
    scaler_price = StandardScaler()
    scaler_target = StandardScaler()

    # Fit scalers on training data only
    scaler_alphas.fit(train_df[alpha_cols].values.astype(float))
    scaler_price.fit(train_df[['close']].values.astype(float))
    scaler_target.fit(train_df[['target']].values.astype(float))

    # Transform all splits
    train_df.loc[:, alpha_cols] = scaler_alphas.transform(train_df[alpha_cols].values.astype(float))
    train_df.loc[:, ['close']] = scaler_price.transform(train_df[['close']].values.astype(float))
    train_df.loc[:, ['target']] = scaler_target.transform(train_df[['target']].values.astype(float))

    val_df.loc[:, alpha_cols] = scaler_alphas.transform(val_df[alpha_cols].values.astype(float))
    val_df.loc[:, ['close']] = scaler_price.transform(val_df[['close']].values.astype(float))
    val_df.loc[:, ['target']] = scaler_target.transform(val_df[['target']].values.astype(float))

    test_df.loc[:, alpha_cols] = scaler_alphas.transform(test_df[alpha_cols].values.astype(float))
    test_df.loc[:, ['close']] = scaler_price.transform(test_df[['close']].values.astype(float))
    test_df.loc[:, ['target']] = scaler_target.transform(test_df[['target']].values.astype(float))

    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create datasets
    train_dataset = StockDataset(train_df, window_size=window_size, dual_stream=True)
    val_dataset = StockDataset(val_df, window_size=window_size, dual_stream=True)
    test_dataset = StockDataset(test_df, window_size=window_size, dual_stream=True)

    # Create dataloaders with adaptive batch size
    batch_size = min(64, max(16, len(train_dataset) // 10))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scalers = {
        'alphas': scaler_alphas,
        'price': scaler_price,
        'target': scaler_target  # Separate scaler for target
    }

    return train_loader, val_loader, test_loader, scalers, num_alphas

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

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, device='cpu'):
    """
    IMPROVED: Train with better hyperparameters
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    print(f"\nTraining on {device}...")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load('best_model.pth'))

    return model, train_losses, val_losses

def evaluate_model(model, test_loader, scalers, device):
    """
    IMPROVED: Evaluate with proper metrics calculation
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

    # Inverse transform predictions and actuals
    predictions_unscaled = scalers['target'].inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_unscaled = scalers['target'].inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Calculate metrics on percentage changes
    mse = np.mean((predictions_unscaled - actuals_unscaled) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_unscaled - actuals_unscaled))

    # MAPE: only calculate where actuals are not near zero
    mask = np.abs(actuals_unscaled) > 0.001
    if mask.sum() > 0:
        mape = np.mean(np.abs((actuals_unscaled[mask] - predictions_unscaled[mask]) / actuals_unscaled[mask])) * 100
    else:
        mape = float('inf')

    # Directional accuracy
    actual_direction = np.sign(actuals_unscaled)
    pred_direction = np.sign(predictions_unscaled)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    # Additional metric: Percentage of predictions within 1% of actual
    within_1pct = np.mean(np.abs(predictions_unscaled - actuals_unscaled) < 0.01) * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional Accuracy': directional_accuracy,
        'Within 1% Accuracy': within_1pct
    }

    return predictions_unscaled, actuals_unscaled, metrics

def plot_predictions(actuals, predictions, ticker, save_path=None):
    """
    Plot actual vs predicted prices
    """
    plt.figure(figsize=(14, 6))

    plt.plot(actuals, label='Actual % Change', linewidth=2, alpha=0.7)
    plt.plot(predictions, label='Predicted % Change', linewidth=2, alpha=0.7)

    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('% Change', fontsize=12)
    plt.title(f'{ticker} Stock Price % Change Prediction', fontsize=14, fontweight='bold')
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

def train_stock_predictor(ticker, company_name, comprehensive_df, alpha_text,
                         window_size=20, num_epochs=100, device='cpu'):
    """
    IMPROVED: Complete pipeline with enhanced model and training
    """
    print("="*80)
    print(f"TRAINING IMPROVED LSTM PREDICTOR FOR {company_name} ({ticker})")
    print("="*80)

    # Prepare data
    train_loader, val_loader, test_loader, scalers, num_alphas = prepare_data_for_training(
        comprehensive_df, ticker, alpha_text, window_size
    )

    if train_loader is None:
        print(f"Failed to prepare data for {ticker}")
        return None, None, None

    # Initialize improved model
    model = ImprovedDualStreamLSTM(
        num_alphas=num_alphas,
        hidden_size=128,
        num_layers=3,
        dropout=0.3
    ).to(device)

    print(f"\nModel Architecture:")
    print(f"  - Alpha stream: {num_alphas} inputs (bidirectional)")
    print(f"  - Price stream: 4 inputs (bidirectional)")
    print(f"  - Hidden size: 128")
    print(f"  - Num layers: 3")
    print(f"  - Attention mechanism: Yes")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=1e-4,
        device=device
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions, actuals, metrics = evaluate_model(
        model, test_loader, scalers, device
    )

    # Print metrics
    print("\n" + "="*80)
    print(f"TEST SET RESULTS FOR {ticker}")
    print("="*80)
    for metric_name, value in metrics.items():
        if value != float('inf'):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: N/A")
    print("="*80)

    # Plot results
    plot_training_history(train_losses, val_losses, ticker,
                         save_path=f'{ticker}_training_history.png')
    plot_predictions(actuals, predictions, ticker,
                    save_path=f'{ticker}_predictions.png')

    return model, metrics, scalers