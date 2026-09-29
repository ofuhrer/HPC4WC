import torch
import torch.nn as nn

class MSE_TemporalDifference_Loss(nn.Module):
    """
    Temporal Difference Loss:
    Penalizes differences between the predicted and true temporal changes.
    
    Args:
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        """
        pred:   Tensor [B, T, H, W] predicted frames
        target: Tensor [B, T, H, W] ground truth frames
        """
        # Temporal differences: frame[t+1] - frame[t]
        pred_diff = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        target_diff = target[:, 1:, :, :] - target[:, :-1, :, :]
        
        # L1 loss between temporal differences
        td_loss = torch.abs(pred_diff - target_diff)

        if self.reduction == 'mean':
            td_loss = td_loss.mean()
        elif self.reduction == 'sum':
            td_loss = td_loss.sum()

         # Base MSE loss
        mse_loss = self.mse(pred, target)

        return mse_loss + 0.5*td_loss