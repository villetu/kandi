import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
energy_net.load_state_dict(torch.load("verkko1.pth"))
energy_net.eval()

def energy(x, y):
    coords = torch.tensor([x, y], dtype=torch.float32)
    return energy_net(coords).item()

def gradient(x, y):
    coords = torch.tensor([x, y], dtype=torch.float32, requires_grad=True)
    E = energy_net(coords)[0]
    E.backward()
    grad = coords.grad
    return -grad[0].item(), -grad[1].item()

# Tilan kehityksen simulointi
x, y = 2, 1     # alkupiste
positions = [(x, y)]
dt = 1

for _ in range(300):
    dx, dy = gradient(x, y)
    x += dx * dt
    y += dy * dt
    positions.append((x, y))
positions = np.array(positions)

# Energiamaasto
X, Y = np.meshgrid(np.linspace(0, 15, 200), np.linspace(0, 15, 200))
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = energy(X[i, j], Y[i, j])
Z = (Z - Z.min()) / (Z.max() - Z.min())

# Animaatio
fig, ax = plt.subplots()
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=0.0, vmax=1.0)
cb = plt.colorbar(contour, ax=ax, label="Normalisoitu energia")
cb.set_ticks(np.arange(0, 1.01, 0.1))

particle, = ax.plot([], [], 'ro', markersize=5)
trail, = ax.plot([], [], 'w-', linewidth=2)

def init():
    particle.set_data([], [])
    trail.set_data([], [])
    return particle, trail

def update(frame):
    particle.set_data([positions[frame, 0]], [positions[frame, 1]])
    trail.set_data([positions[:frame+1, 0]], [positions[:frame+1, 1]])
    return particle, trail

ani = animation.FuncAnimation(fig, update, frames=len(positions),
                              init_func=init, blit=True, interval=40)


plt.xlabel("Hermosolu 1 taajuus (Hz)")
plt.ylabel("Hermosolu 2 taajuus (Hz)")
plt.show()