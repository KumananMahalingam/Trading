"""
Training progress visualization functions
"""
import matplotlib.pyplot as plt


def plot_training_progress(history, ticker, save_path=None):
    """
    Plot comprehensive training progress

    Args:
        history: Dictionary with training history
        ticker: Stock ticker symbol
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(history['train_losses'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_losses'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{ticker} - Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(history['train_accuracies'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Direction Accuracy (%)')
    ax2.set_title(f'{ticker} - Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning rate schedule
    ax3 = axes[1, 0]
    if 'learning_rates' in history:
        ax3.plot(history['learning_rates'], linewidth=2, color='green')
        ax3.set_yscale('log')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title(f'{ticker} - Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)

    # Combined view
    ax4 = axes[1, 1]
    best_epoch = history.get('best_epoch', 0)
    ax4.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
                label=f'Best Epoch: {best_epoch}')

    # Plot both losses on secondary y-axis
    ax4_twin = ax4.twinx()
    ln1 = ax4.plot(history['train_losses'], label='Train Loss', color='blue', linewidth=2)
    ln2 = ax4_twin.plot(history['train_accuracies'], label='Train Acc', color='orange', linewidth=2)

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='blue')
    ax4_twin.set_ylabel('Accuracy (%)', color='orange')
    ax4.set_title(f'{ticker} - Combined Training Progress')

    # Combine legends
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax4.legend(lns, labs, loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training plot to {save_path}")

    plt.show()
    return fig
