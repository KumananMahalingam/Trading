"""
Model ensemble for improved predictions
"""
import torch
import torch.nn as nn


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models for improved performance"""
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)

        # Learnable weights for ensemble
        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def forward(self, alphas, prices_temporal, n_samples=5, training=True):
        predictions = []
        uncertainties = []

        for model in self.models:
            if training:
                pred = model(alphas, prices_temporal, training=training)
                predictions.append(pred)
            else:
                pred, unc = model(alphas, prices_temporal, n_samples=n_samples, training=False)
                predictions.append(pred)
                uncertainties.append(unc)

        predictions = torch.stack(predictions, dim=-1)

        # Weighted average
        weights = torch.softmax(self.weights, dim=0)
        weighted_pred = torch.sum(predictions * weights, dim=-1)

        if not training:
            uncertainties = torch.stack(uncertainties, dim=-1)
            weighted_unc = torch.sum(uncertainties * weights, dim=-1)
            return weighted_pred, weighted_unc

        return weighted_pred