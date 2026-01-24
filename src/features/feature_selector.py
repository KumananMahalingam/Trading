"""
Feature selection utilities
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def select_top_features(df, target_col='target', top_k=30, method='correlation',
                       exclude_cols=None, min_correlation=0.02):
    """
    Select features with highest predictive power for the target

    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        top_k: Number of top features to select
        method: 'correlation', 'mutual_info', or 'both'
        exclude_cols: Columns to exclude from selection
        min_correlation: Minimum correlation threshold

    Returns:
        tuple: (top_features: list, feature_scores: DataFrame)
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'ticker', 'symbol']

    # Get all feature columns
    feature_cols = [col for col in df.columns
                   if col != target_col and col not in exclude_cols]

    # Remove columns with too many NaNs
    valid_cols = []
    for col in feature_cols:
        if df[col].notna().sum() > len(df) * 0.5:
            valid_cols.append(col)

    print(f"\n{'='*80}")
    print(f"FEATURE SELECTION")
    print(f"{'='*80}")
    print(f"Total features available: {len(feature_cols)}")
    print(f"Features with sufficient data: {len(valid_cols)}")

    # Prepare data
    X = df[valid_cols].copy()
    y = df[target_col].copy()

    # Remove rows with NaN in target
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    # Fill remaining NaNs
    X = X.fillna(X.mean())

    feature_scores = pd.DataFrame({'feature': valid_cols})

    # Calculate correlations
    if method in ['correlation', 'both']:
        print(f"\nCalculating correlations...")
        correlations = []
        for col in valid_cols:
            try:
                corr = X[col].corr(y)
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            except:
                correlations.append(0)

        feature_scores['correlation'] = correlations
        feature_scores['abs_correlation'] = feature_scores['correlation'].abs()

    # Calculate mutual information
    if method in ['mutual_info', 'both']:
        print(f"Calculating mutual information...")
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=3)
            feature_scores['mutual_info'] = mi_scores
        except Exception as e:
            print(f"  Warning: Could not calculate mutual info: {e}")
            feature_scores['mutual_info'] = 0

    # Combined score
    if method == 'both':
        feature_scores['correlation_norm'] = (
            feature_scores['abs_correlation'] / feature_scores['abs_correlation'].max()
        )
        feature_scores['mi_norm'] = (
            feature_scores['mutual_info'] / feature_scores['mutual_info'].max()
            if feature_scores['mutual_info'].max() > 0 else 0
        )
        feature_scores['combined_score'] = (
            0.6 * feature_scores['correlation_norm'] +
            0.4 * feature_scores['mi_norm']
        )
        sort_by = 'combined_score'
    elif method == 'correlation':
        sort_by = 'abs_correlation'
    else:
        sort_by = 'mutual_info'

    # Sort and filter
    feature_scores = feature_scores.sort_values(by=sort_by, ascending=False)

    if method in ['correlation', 'both']:
        feature_scores = feature_scores[
            feature_scores['abs_correlation'] >= min_correlation
        ]

    # Select top K
    top_features = feature_scores.head(top_k)['feature'].tolist()

    # Print results
    print(f"\n{'='*80}")
    print(f"TOP {len(top_features)} FEATURES SELECTED")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Feature':<40} {'Corr':<10} {'Score':<10}")
    print(f"{'-'*80}")

    for i, row in feature_scores.head(top_k).iterrows():
        rank = feature_scores.index.get_loc(i) + 1
        feat = row['feature'][:38]
        corr_str = f"{row.get('abs_correlation', 0):.4f}"
        score_str = f"{row.get(sort_by, 0):.4f}"
        print(f"{rank:<6} {feat:<40} {corr_str:<10} {score_str:<10}")

    return top_features, feature_scores