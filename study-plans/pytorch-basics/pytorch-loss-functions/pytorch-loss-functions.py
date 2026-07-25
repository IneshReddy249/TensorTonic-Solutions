import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred = torch.tensor(pred, dtype=torch.float32)

    if method == "mse":
        target_t = torch.tensor(target, dtype=torch.float32)
        return ((pred - target_t) ** 2).mean().item()

    elif method == "cross_entropy":
        target_t = torch.tensor(target, dtype=torch.long)
        max_val = pred.max(dim=1, keepdim=True).values
        shifted = pred - max_val
        log_sum_exp = shifted.exp().sum(dim=1).log() + max_val.squeeze(1)
        correct_logits = pred[torch.arange(pred.shape[0]), target_t]
        return (log_sum_exp - correct_logits).mean().item()

    elif method == "huber":
        target_t = torch.tensor(target, dtype=torch.float32)
        diff = (pred - target_t).abs()
        loss = torch.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
        return loss.mean().item()
