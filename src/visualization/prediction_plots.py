"""
Prediction visualization functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_enhanced_predictions(actuals, predictions, uncertainties, ticker, test_dates, save_path=None):
    """
    Enhanced visualization with uncertainty bands

    Args:
        actuals: Actual return values
        predictions: Predicted return values
        uncertainties: Prediction uncertainties
        ticker: Stock ticker symbol
        test_dates: Array of dates
        save_path: Optional path to save the plot
    """
    # Convert dates
    if isinstance(test_dates[0], str):
        dates = pd.to_datetime(test_dates)
    else:
        dates = test_dates

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Main prediction plot
    ax1 = axes[0, 0]
    ax1.plot(dates, actuals, label='Actual % Change', linewidth=2.5, alpha=0.8,
             color='#2E86DE', marker='o', markersize=6)
    ax1.plot(dates, predictions, label='Predicted % Change', linewidth=2.5, alpha=0.8,
             color='#EE5A6F', marker='s', markersize=6)

    # Add uncertainty bands
    ax1.fill_between(dates, predictions - uncertainties, predictions + uncertainties,
                    alpha=0.3, color='#EE5A6F', label='Uncertainty')

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily % Change')
    ax1.set_title(f'{ticker} - Predictions with Uncertainty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot with perfect prediction line
    ax2 = axes[0, 1]
    ax2.scatter(actuals, predictions, alpha=0.6, c=uncertainties, cmap='viridis')
    ax2.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()],
             'r--', alpha=0.5, label='Perfect Prediction')
    ax2.set_xlabel('Actual % Change')
    ax2.set_ylabel('Predicted % Change')
    ax2.set_title('Prediction vs Actual')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add colorbar for uncertainty
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=uncertainties.min(), vmax=uncertainties.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Uncertainty')

    # Error distribution
    ax3 = axes[1, 0]
    errors = predictions - actuals
    ax3.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.grid(True, alpha=0.3)

    # Uncertainty vs Error
    ax4 = axes[1, 1]
    ax4.scatter(uncertainties, np.abs(errors), alpha=0.6)

    # Add trend line
    if len(errors) > 1:
        z = np.polyfit(uncertainties, np.abs(errors), 1)
        p = np.poly1d(z)
        ax4.plot(np.sort(uncertainties), p(np.sort(uncertainties)), "r--", alpha=0.8)

    ax4.set_xlabel('Uncertainty')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Uncertainty vs Error (should be correlated)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved enhanced plot to {save_path}")

    plt.show()
    return fig
