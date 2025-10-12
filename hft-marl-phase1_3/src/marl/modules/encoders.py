import torch, torch.nn as nn
class CNNTemporalEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU())
    def forward(self, x): return self.net(x)
