# model.py
import torch.nn as nn
import torch.nn.functional as F
import torch

class CVAE(nn.Module):
    def __init__(self, z_dim=20):
        super().__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(28*28 + 10, 400)
        self.fc21 = nn.Linear(400, z_dim)
        self.fc22 = nn.Linear(400, z_dim)
        self.fc3 = nn.Linear(z_dim + 10, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], 1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], 1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar
