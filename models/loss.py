import torch
import torch.nn as nn


class ShiftLoss(nn.Module):
    """
    Time-shifted Pearson loss for handling PPG/video sync misalignment.
    Implements: Loss = 1 - max_τ∈(-δt,δt) Pearson(srPPG(t+τ), sPPG(t))
    """
    def __init__(self, max_shift_sec: float = 0.33, fps: float = 15.0, eps: float = 1e-8):
        super().__init__()
        self.max_shift_sec = max_shift_sec
        self.fps = fps
        self.eps = eps
        self.max_shift_frames = max(1, int(round(max_shift_sec * fps)))

    def forward(self, predicted_ppg: torch.Tensor, target_ppg: torch.Tensor) -> torch.Tensor:
        batch_size = predicted_ppg.shape[0]
        max_corr = torch.zeros(batch_size, device=predicted_ppg.device)

        for shift in range(-self.max_shift_frames, self.max_shift_frames + 1):
            if shift >= 0:
                pred = predicted_ppg[:, shift:]
                targ = target_ppg[:, :-shift] if shift > 0 else target_ppg
            else:
                pred = predicted_ppg[:, :shift]
                targ = target_ppg[:, -shift:]

            if pred.shape[1] == 0:
                continue

            pred_norm = pred - pred.mean(dim=1, keepdim=True)
            targ_norm = targ - targ.mean(dim=1, keepdim=True)

            numerator = (pred_norm * targ_norm).sum(dim=1)
            denominator = (
                pred_norm.pow(2).sum(dim=1).sqrt() *
                targ_norm.pow(2).sum(dim=1).sqrt() +
                self.eps
            )
            corr = numerator / denominator
            max_corr = torch.maximum(max_corr, corr)

        return (1.0 - max_corr).mean()


class CNNLoss(nn.Module):
    def __init__(self, spectral_alpha: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.spectral_alpha = spectral_alpha
        self.eps = eps

    def forward(self, predicted_ppg: torch.Tensor, target_ppg: torch.Tensor) -> torch.Tensor:
        pearson_loss = self.negative_pearson(predicted_ppg, target_ppg)
        if self.spectral_alpha <= 0:
            return pearson_loss
        return pearson_loss + self.spectral_alpha * self.spectral_loss(predicted_ppg, target_ppg)

    def negative_pearson(self, predicted_ppg: torch.Tensor, target_ppg: torch.Tensor) -> torch.Tensor:
        predicted_ppg = predicted_ppg - predicted_ppg.mean(dim=1, keepdim=True)
        target_ppg = target_ppg - target_ppg.mean(dim=1, keepdim=True)

        numerator = torch.sum(predicted_ppg * target_ppg, dim=1)
        denominator = torch.sqrt(
            torch.sum(predicted_ppg.pow(2), dim=1)
            * torch.sum(target_ppg.pow(2), dim=1)
            + self.eps
        )
        correlation = numerator / denominator
        return torch.mean(1 - correlation)

    def spectral_loss(self, predicted_ppg: torch.Tensor, target_ppg: torch.Tensor) -> torch.Tensor:
        predicted_spectrum = torch.abs(torch.fft.rfft(predicted_ppg, dim=1))
        target_spectrum = torch.abs(torch.fft.rfft(target_ppg, dim=1))

        predicted_spectrum = predicted_spectrum / (predicted_spectrum.sum(dim=1, keepdim=True) + self.eps)
        target_spectrum = target_spectrum / (target_spectrum.sum(dim=1, keepdim=True) + self.eps)

        return torch.mean(torch.abs(predicted_spectrum - target_spectrum))
