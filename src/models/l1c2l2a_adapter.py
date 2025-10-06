import torch
import torch.nn as nn
import torch.nn.functional as F

class L1C2L2AAdapter(nn.Module):
    """Linear transform in CLIP embedding space (D×D)."""
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,P,D]  or  [B*P,D]
        x = self.linear(x)
        x = self.dropout(x)
        return x
