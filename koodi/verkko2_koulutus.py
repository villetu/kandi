import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Neuroverkko kahdelle attraktorille
class EnergyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, coords):
        return self.net(coords)

energy_net = EnergyNet()

# Data koulutusta varten
N = 5000
x = np.random.uniform(0, 15, N)
y = np.random.uniform(0, 15, N)
X_train = np.stack([x, y], axis=1)

# Kaksi attraktoria, keskipisteet (5,7) ja (9,4)
background = 0.1 * np.cos(0.5 * x) + 0.1 * np.cos(0.5 * y)
attractor1 = -np.exp(-((x - 5)**2 + (y - 7)**2))
attractor2 = -np.exp(-((x - 9)**2 + (y - 4)**2))
target_E = background + attractor1 + attractor2
y_train = target_E.reshape(-1, 1)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Verkon koulutus
optimizer = optim.Adam(energy_net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    pred = energy_net(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(energy_net.state_dict(), "verkko2.pth")