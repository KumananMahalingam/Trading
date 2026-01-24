"""
PyTorch Dataset with data augmentation
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.interpolate import CubicSpline


class EnhancedStockDataset(Dataset):
    """
    Dataset with data augmentation and advanced features
    """
    def __init__(self, data, window_size=20, augment=True, dual_stream=True):
        super().__init__()
        self.window_size = window_size
        self.augment = augment
        self.dual_stream = dual_stream

        # Extract features
        alpha_cols = [col for col in data.columns if col.startswith('alpha_')]
        price_col = 'close'
        temporal_cols = ['day_of_week', 'day_of_month', 'month']

        # Extract price features (open, high, low, volume)
        price_features = ['open', 'high', 'low', 'volume']
        price_features = [col for col in price_features if col in data.columns]

        self.alphas = data[alpha_cols].values.astype(np.float32)
        self.prices = data[price_features].values.astype(np.float32)
        self.temporal = data[temporal_cols].values.astype(np.float32)
        self.targets = data['target'].values.astype(np.float32)
        self.dates = data['date'].values if 'date' in data.columns else None

        # Additional features for augmentation
        self.returns = data['target'].values.astype(np.float32) if 'target' in data.columns else None

        # Create sliding windows
        self.samples = []
        for i in range(len(data) - window_size):
            self.samples.append(i)

        print(f"  Created {len(self.samples)} samples with window_size={window_size}")

    def __len__(self):
        return len(self.samples)

    def augment_sequence(self, sequence, augmentation_type='all'):
        """Apply data augmentation to a sequence"""
        if not self.augment:
            return sequence

        if augmentation_type == 'all':
            augmentation_type = np.random.choice(['noise', 'scale', 'time_warp'])

        if augmentation_type == 'noise':
            # Add correlated noise (like Brownian motion)
            noise_level = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, sequence.shape)
            noise = np.cumsum(noise, axis=0)
            return sequence + noise

        elif augmentation_type == 'scale':
            # Scale the entire sequence
            scale = np.random.uniform(0.95, 1.05)
            return sequence * scale

        elif augmentation_type == 'time_warp':
            # Time warping (stretching/squeezing)
            try:
                orig_time = np.arange(len(sequence))
                warp_points = np.random.normal(0, 0.1, 3).cumsum()
                warp_time = orig_time + warp_points * (len(sequence) / 10)
                cs = CubicSpline(warp_time, sequence, axis=0)
                return cs(orig_time)
            except:
                return sequence

        return sequence

    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        end_idx = start_idx + self.window_size

        if self.dual_stream:
            # Alpha stream
            alphas_window = self.alphas[start_idx:end_idx].copy()

            # Price stream (close + temporal)
            price_window = self.prices[start_idx:end_idx, 0:1].copy()  # Only close price
            temporal_window = self.temporal[start_idx:end_idx].copy()

            # Apply augmentation during training
            if self.augment and idx % 3 == 0:  # Augment 1/3 of samples
                alphas_window = self.augment_sequence(alphas_window)
                price_window = self.augment_sequence(price_window)

            prices_temporal_window = np.concatenate([price_window, temporal_window], axis=1)

            target = self.targets[end_idx]

            return (
                torch.FloatTensor(alphas_window),
                torch.FloatTensor(prices_temporal_window),
                torch.FloatTensor([target])
            )
        else:
            # Single stream (all features concatenated)
            all_features = np.concatenate([
                self.alphas[start_idx:end_idx],
                self.prices[start_idx:end_idx],
                self.temporal[start_idx:end_idx]
            ], axis=1)

            # Apply augmentation
            if self.augment and idx % 3 == 0:
                all_features = self.augment_sequence(all_features)

            target = self.targets[end_idx]

            return (
                torch.FloatTensor(all_features),
                torch.FloatTensor([target])
            )