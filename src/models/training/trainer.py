"""
Model training functions
NOTE: This is Part 1 of 2 - contains prepare_data and training epoch functions
Part 2 will contain train_stock_predictor and train_ensemble
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.models.training.dataset import EnhancedStockDataset
from src.models.training.losses import create_balanced_hybrid_loss
from src.features.alpha_computer import EnhancedAlphaComputer
from src.features.feature_engineering import add_advanced_features, create_target_safely
from src.features.feature_selector import select_top_features
from src.models.architectures.dual_stream_lstm import ImprovedDualStreamLSTM
from src.models.architectures.ensemble import ModelEnsemble
from src.models.evaluation.evaluator import evaluate_enhanced_model, evaluate_enhanced_model_with_viz
from src.visualization.training_plots import plot_training_progress
from src.visualization.prediction_plots import plot_enhanced_predictions
from src.visualization.stock_price_plots import plot_stock_price_predictions


def prepare_data_with_fixes(df, ticker, alpha_text, window_size=30,
                            use_feature_selection=True, top_k=30):
    """
    Enhanced data preparation with all fixes applied
    """
    print(f"\n{'='*80}")
    print(f"PREPARING DATA FOR {ticker} (WITH FIXES)")
    print(f"{'='*80}")

    # Add temporal features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek / 6.0
        df['day_of_month'] = df['date'].dt.day / 31.0
        df['month'] = df['date'].dt.month / 12.0

    # Add advanced features
    df = add_advanced_features(df)

    # Create target SAFELY (no data leakage)
    df = create_target_safely(df)

    # Compute alphas
    alpha_computer = EnhancedAlphaComputer(df)
    df, num_alphas = alpha_computer.add_alphas_from_text(alpha_text, max_alphas=5)

    selected_features = None
    feature_scores = None

    if use_feature_selection:
        print(f"\n{'='*80}")
        print("APPLYING FEATURE SELECTION")
        print(f"{'='*80}")

        # Select top features
        selected_features, feature_scores = select_top_features(
            df=df,
            target_col='target',
            top_k=top_k,
            method='both',
            exclude_cols=['date', 'ticker'],
            min_correlation=0.01
        )

        # Always include alpha features
        alpha_features = [col for col in df.columns if col.startswith('alpha_')]
        for alpha_feat in alpha_features:
            if alpha_feat not in selected_features:
                selected_features.append(alpha_feat)

        # Always include temporal features
        temporal_features = ['day_of_week', 'day_of_month', 'month']
        temporal_features = [f for f in temporal_features if f in df.columns]
        for temp_feat in temporal_features:
            if temp_feat not in selected_features:
                selected_features.append(temp_feat)

        # Always include price features for dual stream
        price_features = ['open', 'high', 'low', 'close', 'volume']
        price_features = [f for f in price_features if f in df.columns]
        for price_feat in price_features:
            if price_feat not in selected_features:
                selected_features.append(price_feat)

        print(f"\n✓ Final feature set: {len(selected_features)} features")
        print(f"  - Alpha features: {len([f for f in selected_features if 'alpha_' in f])}")
        print(f"  - Temporal features: {len([f for f in selected_features if f in temporal_features])}")
        print(f"  - Price features: {len([f for f in selected_features if f in price_features])}")

        # Keep only selected features + target + date
        keep_cols = selected_features + ['target']
        if 'date' in df.columns:
            keep_cols.append('date')

        keep_cols = list(dict.fromkeys(keep_cols))
        df = df[keep_cols].copy()

    # Check class balance BEFORE splitting
    if 'target' in df.columns:
        up_pct = (df['target'] > 0).sum() / len(df) * 100
        down_pct = (df['target'] < 0).sum() / len(df) * 100
        neutral_pct = (df['target'] == 0).sum() / len(df) * 100

        print(f"\n📊 Target Distribution (BEFORE split):")
        print(f"  UP: {up_pct:.1f}%")
        print(f"  DOWN: {down_pct:.1f}%")
        print(f"  NEUTRAL: {neutral_pct:.1f}%")

        if up_pct > 70 or down_pct > 70:
            print(f"⚠️ WARNING: Severe class imbalance detected!")

    # Handle missing values
    for col in df.columns:
        if col not in ['date', 'target']:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].ffill().bfill().fillna(0)

    # Drop rows with NaN target
    df = df.dropna(subset=['target'])

    if len(df) < window_size + 50:
        print(f"❌ ERROR: Insufficient data")
        return None, None, None, None, 0, None, None

    # Create binary labels for stratification
    df['target_class'] = (df['target'] > 0).astype(int)

    # Stratified split
    train_val_indices, test_indices = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df['target_class'],
        random_state=42
    )

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.20,
        stratify=df.iloc[train_val_indices]['target_class'],
        random_state=42
    )

    # Sort indices to maintain temporal order
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)

    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # Remove temporary column
    for split_df in [train_df, val_df, test_df]:
        if 'target_class' in split_df.columns:
            split_df.drop('target_class', axis=1, inplace=True)

    print(f"\n✓ Stratified split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Check class distribution in each split
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        up_pct = (split_df['target'] > 0).sum() / len(split_df) * 100
        down_pct = (split_df['target'] < 0).sum() / len(split_df) * 100
        print(f"  {name} - UP: {up_pct:.1f}%, DOWN: {down_pct:.1f}%")

    # Scale features
    alpha_cols = [col for col in df.columns if col.startswith('alpha_')]
    price_cols = ['open', 'high', 'low', 'close', 'volume']

    scalers = {}

    for group_name, group_cols in [('alphas', alpha_cols), ('price', price_cols), ('target', ['target'])]:
        existing_cols = [col for col in group_cols if col in train_df.columns]
        if not existing_cols:
            continue

        scaler = StandardScaler()
        train_df[existing_cols] = scaler.fit_transform(train_df[existing_cols])
        val_df[existing_cols] = scaler.transform(val_df[existing_cols])
        test_df[existing_cols] = scaler.transform(test_df[existing_cols])

        scalers[group_name] = scaler

    # Create datasets
    train_dataset = EnhancedStockDataset(train_df, window_size=window_size, augment=True)
    val_dataset = EnhancedStockDataset(val_df, window_size=window_size, augment=False)
    test_dataset = EnhancedStockDataset(test_df, window_size=window_size, augment=False)

    # Adaptive batch size
    batch_size = min(64, max(16, len(train_dataset) // 10))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_dates = test_df['date'].values if 'date' in test_df.columns else None

    print(f"\n✓ Data preparation complete with fixes applied!")

    return train_loader, val_loader, test_loader, scalers, num_alphas, test_dates, train_df


def train_epoch_enhanced(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Enhanced training epoch with gradient clipping"""
    model.train()
    total_loss = 0
    total_direction_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        alphas, prices_temporal, targets = batch
        alphas = alphas.to(device)
        prices_temporal = prices_temporal.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(alphas, prices_temporal)
        loss = criterion(predictions, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate direction accuracy
        pred_sign = torch.sign(predictions).detach()
        true_sign = torch.sign(targets).detach()
        direction_correct = (pred_sign == true_sign).float().sum().item()

        total_loss += loss.item()
        total_direction_correct += direction_correct
        total_samples += len(targets)

        if scheduler and hasattr(scheduler, 'batch_step'):
            scheduler.batch_step()

    avg_loss = total_loss / len(train_loader)
    direction_accuracy = (total_direction_correct / total_samples) * 100

    return avg_loss, direction_accuracy


def validate_enhanced(model, val_loader, criterion, device, n_samples=5):
    """Enhanced validation with uncertainty estimation"""
    model.eval()
    total_loss = 0
    total_direction_correct = 0
    total_samples = 0

    all_predictions = []
    all_targets = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in val_loader:
            alphas, prices_temporal, targets = batch
            alphas = alphas.to(device)
            prices_temporal = prices_temporal.to(device)
            targets = targets.to(device)

            # Skip batches with size 1
            if alphas.shape[0] == 1:
                continue

            predictions, uncertainties = model(alphas, prices_temporal, n_samples=n_samples, training=False)

            loss = criterion(predictions, targets)

            pred_sign = torch.sign(predictions)
            true_sign = torch.sign(targets)
            direction_correct = (pred_sign == true_sign).float().sum().item()

            total_loss += loss.item()
            total_direction_correct += direction_correct
            total_samples += len(targets)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_uncertainties.append(uncertainties.cpu().numpy())

    if total_samples == 0:
        return {
            'loss': float('inf'),
            'direction_accuracy': 0,
            'mse': float('inf'),
            'mae': float('inf'),
            'uncertainty_corr': 0
        }, np.array([]), np.array([]), np.array([])

    avg_loss = total_loss / len(val_loader)
    direction_accuracy = (total_direction_correct / total_samples) * 100

    all_predictions = np.concatenate(all_predictions).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_uncertainties = np.concatenate(all_uncertainties).flatten()

    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))

    errors = np.abs(all_predictions - all_targets)
    uncertainty_corr = np.corrcoef(errors, all_uncertainties)[0, 1] if len(errors) > 1 else 0

    metrics = {
        'loss': avg_loss,
        'direction_accuracy': direction_accuracy,
        'mse': mse,
        'mae': mae,
        'uncertainty_corr': uncertainty_corr
    }

    return metrics, all_predictions, all_targets, all_uncertainties

def train_with_fixes(model, train_loader, val_loader, num_epochs=150,
                     learning_rate=3e-4, device='cpu', train_df=None):
    """
    Training with proper class balancing and regularization
    """
    print(f"\n{'='*80}")
    print("TRAINING WITH FIXES")
    print(f"{'='*80}")

    # Create balanced loss
    criterion = create_balanced_hybrid_loss(train_df)

    # Optimizer with stronger weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-3,
        betas=(0.9, 0.999)
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    patience = 30

    # Track best balanced metric separately
    best_balanced_metric = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    # Track UP/DOWN precision separately
    train_up_precisions = []
    train_down_precisions = []
    val_up_precisions = []
    val_down_precisions = []

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        up_correct = 0
        up_total = 0
        down_correct = 0
        down_total = 0

        for batch in train_loader:
            alphas, prices_temporal, targets = batch
            alphas = alphas.to(device)
            prices_temporal = prices_temporal.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(alphas, prices_temporal, training=True)
            loss = criterion(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate metrics
            pred_sign = torch.sign(predictions).detach()
            true_sign = torch.sign(targets).detach()

            correct = (pred_sign == true_sign).float()
            total_correct += correct.sum().item()
            total_samples += len(targets)

            # Track UP/DOWN separately
            up_mask = true_sign > 0
            down_mask = true_sign < 0

            if up_mask.any():
                up_correct += (pred_sign[up_mask] > 0).float().sum().item()
                up_total += up_mask.sum().item()

            if down_mask.any():
                down_correct += (pred_sign[down_mask] < 0).float().sum().item()
                down_total += down_mask.sum().item()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_acc = (total_correct / total_samples) * 100
        train_up_prec = (up_correct / up_total * 100) if up_total > 0 else 0
        train_down_prec = (down_correct / down_total * 100) if down_total > 0 else 0

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_up_precisions.append(train_up_prec)
        train_down_precisions.append(train_down_prec)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        val_up_correct = 0
        val_up_total = 0
        val_down_correct = 0
        val_down_total = 0

        with torch.no_grad():
            for batch in val_loader:
                alphas, prices_temporal, targets = batch
                alphas = alphas.to(device)
                prices_temporal = prices_temporal.to(device)
                targets = targets.to(device)

                predictions = model(alphas, prices_temporal, training=True)
                loss = criterion(predictions, targets)

                pred_sign = torch.sign(predictions)
                true_sign = torch.sign(targets)

                correct = (pred_sign == true_sign).float()
                val_correct += correct.sum().item()
                val_samples += len(targets)

                up_mask = true_sign > 0
                down_mask = true_sign < 0

                if up_mask.any():
                    val_up_correct += (pred_sign[up_mask] > 0).float().sum().item()
                    val_up_total += up_mask.sum().item()

                if down_mask.any():
                    val_down_correct += (pred_sign[down_mask] < 0).float().sum().item()
                    val_down_total += down_mask.sum().item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_acc = (val_correct / val_samples) * 100
        val_up_prec = (val_up_correct / val_up_total * 100) if val_up_total > 0 else 0
        val_down_prec = (val_down_correct / val_down_total * 100) if val_down_total > 0 else 0

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_up_precisions.append(val_up_prec)
        val_down_precisions.append(val_down_prec)

        # Step scheduler
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Print progress with UP/DOWN precision
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}]")
            print(f"  Train - Loss: {train_loss:.6f}, Acc: {train_acc:.1f}%, "
                  f"UP: {train_up_prec:.1f}%, DOWN: {train_down_prec:.1f}%")
            print(f"  Val   - Loss: {val_loss:.6f}, Acc: {val_acc:.1f}%, "
                  f"UP: {val_up_prec:.1f}%, DOWN: {val_down_prec:.1f}%")

        # Calculate balanced metric
        balanced_metric = min(val_up_prec, val_down_prec)

        # Save model if:
        # 1. Loss is improving AND balanced metric > 20%
        # OR
        # 2. Balanced metric is best so far (even if loss is worse)

        save_model = False

        if val_loss < best_val_loss and balanced_metric >= 20:
            # Both loss and balance are good
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            save_model = True
            print(f"  ✓ Saving: Better loss ({val_loss:.6f}) + balanced ({balanced_metric:.1f}%)")

        elif balanced_metric > best_balanced_metric and balanced_metric >= 25:
            # Best balance so far
            best_balanced_metric = balanced_metric
            best_val_acc = val_acc
            patience_counter = 0
            save_model = True
            print(f"  ✓ Saving: Best balance ({balanced_metric:.1f}%)")

        else:
            patience_counter += 1

        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_up_prec': val_up_prec,
                'val_down_prec': val_down_prec,
                'balanced_metric': balanced_metric,
            }, 'best_model_checkpoint.pth')

            torch.save(model.state_dict(), 'best_model.pth')

        # Stop if no improvement for patience epochs
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            print(f"  Best validation accuracy: {best_val_acc:.1f}%")
            break

    # Load best model
    checkpoint = torch.load('best_model_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\n✓ Training complete!")
    print(f"  Best val accuracy: {checkpoint['val_acc']:.1f}%")
    print(f"  Best UP precision: {checkpoint['val_up_prec']:.1f}%")
    print(f"  Best DOWN precision: {checkpoint['val_down_prec']:.1f}%")
    print(f"  Balanced metric: {checkpoint.get('balanced_metric', 0):.1f}%")

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'train_up_precisions': train_up_precisions,
        'train_down_precisions': train_down_precisions,
        'val_up_precisions': val_up_precisions,
        'val_down_precisions': val_down_precisions,
        'best_epoch': checkpoint['epoch']
    }

def train_stock_predictor(ticker, company_name, comprehensive_df, alpha_text, num_epochs=150, device='cpu'):
    """
    Main function to train enhanced stock predictor
    """
    print(f"\n{'='*80}")
    print(f"ENHANCED TRAINING FOR {company_name} ({ticker})")
    print(f"{'='*80}")

    # Prepare enhanced data
    train_loader, val_loader, test_loader, scalers, num_alphas, test_dates, train_df = \
        prepare_data_with_fixes(
            df=comprehensive_df,
            ticker=ticker,
            alpha_text=alpha_text,
            window_size=30,
            use_feature_selection=True,  # Enable feature selection
            top_k=30  # Select top 30 features
        )

    if train_loader is None:
        print(f" Data preparation failed for {ticker}")
        return None, None, None

    # Initialize enhanced model
    model = ImprovedDualStreamLSTM(
        num_alphas=num_alphas,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        num_heads=4
    ).to(device)

    print(f"\nModel Architecture:")
    print(f"  - Alpha stream: {num_alphas} features")
    print(f"  - Hidden size: {model.hidden_size}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {device}")

    model, training_history = train_with_fixes(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=3e-4,
        device=device,
        train_df=train_df
    )

    # Evaluate on test set
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}")

    predictions, actuals, uncertainties, metrics = evaluate_enhanced_model(
        model=model,
        test_loader=test_loader,
        scalers=scalers,
        device=device,
        n_samples=10
    )

    # Print metrics
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  Directional Accuracy: {metrics['Directional Accuracy']:.2f}%")
    print(f"  Up Precision: {metrics['Up Precision']:.2f}%")
    print(f"  Down Precision: {metrics['Down Precision']:.2f}%")
    print(f"  Large Move Hit Rate: {metrics['Large Move Hit Rate']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    print(f"  Win Rate: {metrics['Win Rate']:.2f}%")
    print(f"  Avg Uncertainty: {metrics['Avg Uncertainty']:.6f}")
    print(f"  Uncertainty Correlation: {metrics['Uncertainty Correlation']:.3f}")

    # Visualize results
    print(f"\nGenerating visualizations...")

    # Plot training progress
    plot_training_progress(training_history, ticker, save_path=f"{ticker}_training_progress.png")

    # Plot predictions with uncertainty
    plot_enhanced_predictions(
        actuals, predictions, uncertainties, ticker, test_dates,
        save_path=f"{ticker}_predictions_enhanced.png"
    )

    # Feature importance analysis (optional)
    print(f"\nFeature Analysis:")

    # Get alpha feature importance
    alpha_cols = [col for col in comprehensive_df.columns if col.startswith('alpha_')]
    if alpha_cols:
        print(f"  Alpha features: {', '.join(alpha_cols)}")

        # Simple correlation analysis
        for alpha_col in alpha_cols[:5]:  # Show top 5
            if alpha_col in comprehensive_df.columns:
                corr = comprehensive_df[alpha_col].corr(comprehensive_df['target'])
                print(f"    {alpha_col}: correlation with target = {corr:.3f}")

    return model, metrics, scalers

def train_ensemble(ticker, company_name, comprehensive_df, alpha_text,
                  n_models=3, window_size=30, device='cpu'):
    """
    Train an ensemble of models for improved robustness
    """
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL ENSEMBLE ({n_models} models)")
    print(f"{'='*80}")

    models = []

    # Variables to store for visualization
    test_df = None
    val_df = None
    test_loader = None
    val_loader = None
    scalers = None

    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}...")

        result = prepare_data_with_fixes(
            comprehensive_df, ticker, alpha_text,
            window_size=window_size,
            use_feature_selection=True,
            top_k=30
        )

        if result[0] is None:
            continue

        train_loader, val_loader, test_loader, scalers, num_alphas, test_dates, train_df = result

        test_size = len(test_loader.dataset) + window_size if hasattr(test_loader, 'dataset') else 50
        val_size = len(val_loader.dataset) + window_size if hasattr(val_loader, 'dataset') else 50
        test_df = comprehensive_df.iloc[-(test_size):].copy()
        val_df = comprehensive_df.iloc[-(test_size + val_size):-test_size].copy()

        # Initialize model
        model = ImprovedDualStreamLSTM(
            num_alphas=num_alphas,
            hidden_size=128,
            num_layers=3,
            dropout=0.3 + i*0.05,  # Vary dropout
            num_heads=4
        ).to(device)

        # Train with different learning rate
        lr = 3e-4 * (0.8 ** i)  # Decaying learning rates
        model, _ = train_with_fixes(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,  # Fewer epochs per model
            learning_rate=lr,
            device=device,
            train_df=train_df
        )

        models.append(model)

    # Create ensemble
    if models:
        ensemble = ModelEnsemble(models).to(device)
        print(f"\n Ensemble created with {len(models)} models")

        # Determine which data to use for evaluation
        if test_loader is None or len(test_loader.dataset) == 0:
            print("\n Test set is empty, using validation set for evaluation")
            eval_loader = val_loader
            eval_df = val_df
        else:
            eval_loader = test_loader
            eval_df = test_df

        # Use new evaluation function with visualization support
        predictions, actuals, uncertainties, metrics, dates, initial_price = evaluate_enhanced_model_with_viz(
            model=ensemble,
            test_loader=eval_loader,
            test_df=eval_df,
            scalers=scalers,
            ticker=ticker,
            device=device,
            n_samples=10
        )

        # Only print and visualize if we have predictions
        if len(predictions) > 0:
            print(f"\nEnsemble Performance:")
            print(f"  Directional Accuracy: {metrics['Directional Accuracy']:.2f}%")
            print(f"  RMSE: {metrics['RMSE']:.6f}")

            if initial_price is not None and dates is not None:
                print(f"\n Creating stock price visualization...")

                plot_stock_price_predictions(
                    actuals=actuals,
                    predictions=predictions,
                    uncertainties=uncertainties,
                    ticker=ticker,
                    test_dates=dates,
                    initial_price=initial_price,
                    save_path=f'{ticker}_stock_price_prediction.png'
                )
        else:
            print(f"\n  No predictions made")
            metrics = None

        return ensemble, metrics, scalers

    return None, None, None