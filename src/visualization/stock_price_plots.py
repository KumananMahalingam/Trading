"""
Stock price visualization functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_stock_price_predictions(actuals, predictions, uncertainties, ticker,
                                 test_dates, initial_price, save_path=None):
    """
    Create comprehensive stock price visualization with actual vs predicted prices

    Args:
        actuals: Actual return values
        predictions: Predicted return values
        uncertainties: Prediction uncertainties
        ticker: Stock ticker symbol
        test_dates: Array of dates
        initial_price: Initial stock price
        save_path: Optional path to save the plot
    """
    # Convert dates
    if isinstance(test_dates[0], str):
        dates = pd.to_datetime(test_dates)
    else:
        dates = test_dates

    # Convert % changes to actual prices
    actual_prices = [initial_price]
    predicted_prices = [initial_price]

    for i in range(len(actuals)):
        actual_next = actual_prices[-1] * (1 + actuals[i])
        predicted_next = predicted_prices[-1] * (1 + predictions[i])
        actual_prices.append(actual_next)
        predicted_prices.append(predicted_next)

    actual_prices = actual_prices[1:]
    predicted_prices = predicted_prices[1:]

    # Calculate prediction bands
    predicted_upper = []
    predicted_lower = []
    running_price = initial_price

    for i in range(len(predictions)):
        upper_return = predictions[i] + uncertainties[i]
        lower_return = predictions[i] - uncertainties[i]
        upper_price = running_price * (1 + upper_return)
        lower_price = running_price * (1 + lower_return)
        predicted_upper.append(upper_price)
        predicted_lower.append(lower_price)
        running_price = running_price * (1 + predictions[i])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # Actual vs Predicted Stock Prices
    ax1 = axes[0, 0]
    ax1.plot(dates, actual_prices, label='Actual Stock Price',
             linewidth=3.5, alpha=0.9, color='#2E86DE', marker='o', markersize=8)
    ax1.plot(dates, predicted_prices, label='Predicted Stock Price',
             linewidth=3.5, alpha=0.9, color='#EE5A6F', marker='s', markersize=8, linestyle='--')

    if len(predicted_upper) > 0 and np.max(uncertainties) > 0:
        ax1.fill_between(dates, predicted_lower, predicted_upper,
                         alpha=0.2, color='#EE5A6F', label='95% Confidence')

    ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Stock Price ($)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{ticker} - Actual vs Predicted Stock Prices',
                  fontsize=18, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=12, loc='best', framealpha=0.9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Daily Returns
    ax2 = axes[0, 1]
    x = np.arange(len(actuals))
    width = 0.35
    ax2.bar(x - width/2, actuals * 100, width, label='Actual Returns', alpha=0.8, color='#2E86DE')
    ax2.bar(x + width/2, predictions * 100, width, label='Predicted Returns', alpha=0.8, color='#EE5A6F')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Daily Return (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Daily Returns Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Cumulative Returns
    ax3 = axes[1, 0]
    actual_cumulative = np.cumprod(1 + actuals) - 1
    predicted_cumulative = np.cumprod(1 + predictions) - 1
    ax3.plot(dates, actual_cumulative * 100, label='Actual', linewidth=2.5, color='#2E86DE', marker='o')
    ax3.plot(dates, predicted_cumulative * 100, label='Predicted', linewidth=2.5, color='#EE5A6F', marker='s', linestyle='--')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Performance Metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    final_actual_return = (actual_prices[-1] / initial_price - 1) * 100
    final_predicted_return = (predicted_prices[-1] / initial_price - 1) * 100
    price_error = abs(actual_prices[-1] - predicted_prices[-1])
    price_error_pct = (price_error / actual_prices[-1]) * 100
    directional_accuracy = np.mean(np.sign(actuals) == np.sign(predictions)) * 100

    metrics_text = f"""
📊 PERFORMANCE METRICS
{'='*40}

Initial Price:        ${initial_price:.2f}
Final Actual Price:   ${actual_prices[-1]:.2f}
Final Predicted:      ${predicted_prices[-1]:.2f}

Price Error:          ${price_error:.2f} ({price_error_pct:.2f}%)

Actual Return:        {final_actual_return:+.2f}%
Predicted Return:     {final_predicted_return:+.2f}%

Directional Accuracy: {directional_accuracy:.1f}%
"""

    ax4.text(0.1, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray',
                      alpha=0.8, edgecolor='black', linewidth=2))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved stock price visualization to {save_path}")

    plt.show()
    return fig
