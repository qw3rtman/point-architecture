import torch

class SpatialSoftmax(torch.nn.Module):
    """
    IMPORTANT:
    i in [0, 1], where 0 is at the bottom, 1 is at the top
    j in [-1, 1]

    (  1, -1) ... (  1,   1) ... (  1, 1)
              ...            ...
    (0.5, -1) ... (0.5, 0.5) ... (0.5, 1)
              ...            ...
    (  0, -1) ... (  0, 0.5) ... (  0, 1)
    ...
    """
    def __init__(self, temperature=1.0):
        super().__init__()

        self.temperature = temperature

    def forward(self, logit):
        """
        Assumes logits is size (n, c, h, w)
        """
        flat = logit.view(logit.shape[:-2] + (-1,))
        weights = torch.nn.functional.softmax(flat / self.temperature, dim=-1).view_as(logit)

        x = (weights.sum(-2) * torch.linspace(-1, 1, logit.shape[-1]).to(logit.device)).sum(-1)
        y = (weights.sum(-1) * torch.linspace(-1, 1, logit.shape[-2]).to(logit.device)).sum(-1)

        return torch.stack((x, y), -1)
