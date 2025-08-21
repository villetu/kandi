import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt = 0.05
T1 = 5
T2 = 40
n1 = int(T1 / dt)
n2 = int(T2 / dt)

target_radius = 0.5
damping = 0.1
angular_velocity = 0.5

x0, y0 = 7, 9.2
x1, y1 = 5, 8.95
xs, ys = [], []

# Suora liike
for i in range(n1):
    step = i / n1
    x = (1 - step) * x0 + step * x1
    y = (1 - step) * y0 + step * y1
    xs.append(x)
    ys.append(y)

# Kiertorata, kesipiste (5,7)
x, y = x1 - 5, y1 - 7
for j in range(n2):
    r = np.sqrt(x**2 + y**2)
    r += -damping * (r - target_radius) * dt

    angle = np.arctan2(y, x)
    angle += angular_velocity * dt
    x = r * np.cos(angle)
    y = r * np.sin(angle)

    xs.append(x + 5)
    ys.append(y + 7)

# Energiamaasto
grid_size = 300
xlim = ylim = (0, 10)
xlim_graph = (3,8)
ylim_graph = (5,10)
x_vals = np.linspace(*xlim, grid_size)
y_vals = np.linspace(*ylim, grid_size)
X, Y = np.meshgrid(x_vals, y_vals)
R = np.sqrt((X - 5)**2 + (Y - 3)**2)

def compute_energy(r, v):
    potential = 0.5 * (r - target_radius)**2
    kinetic = 0.5 * (r * v)**2
    return potential + kinetic

energy_field = compute_energy(R, angular_velocity)

mask_outer = R > 2     # energia vakio attraktorin ulkopuolella
energy_field[mask_outer] = energy_field[R <= 2].max()

energy_norm = (energy_field - energy_field.min()) / (energy_field.max() - energy_field.min())

# Animaatio
fig, ax = plt.subplots()
ax.grid()
ax.set_facecolor('#fce803')
ax.set_xlim(xlim_graph)
ax.set_ylim(ylim_graph)
im = ax.imshow(energy_norm, extent=(*xlim, *ylim))
cb = plt.colorbar(im, ax=ax, label="Normalisoitu energia")
cb.set_ticks(np.arange(0, 1.01, 0.1))

particle, = ax.plot([], [], 'ro', markersize=5)
trail, = ax.plot([], [], 'w-', linewidth=2)

def update(frame):
    trail.set_data(xs[:frame], ys[:frame])
    particle.set_data([xs[frame-1]], [ys[frame-1]])
    return trail, particle

ani = animation.FuncAnimation(fig, update, frames=len(xs), blit=True, interval=20)

plt.xlabel("Hermosolu 1 taajuus (Hz)")
plt.ylabel("Hermosolu 2 taajuus (Hz)")
plt.show()