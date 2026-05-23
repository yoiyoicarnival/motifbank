import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ======================================
# RG PARAMETERS
# ======================================

# Sierpinski mode
r = 2.847
theta_deg = 12.95

det_sign = 1

# Vicsek example:
# r = 1.163
# theta_deg = 47.89
# det_sign = 1

# T-fractal example:
# r = 0.840
# theta_deg = 0.0
# det_sign = -1

# ======================================
# VIDEO SETTINGS
# ======================================

FRAMES = 240
POINTS = 350
OUTPUT_DIR = "frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)


theta = np.radians(theta_deg)


# ======================================
# RG EVOLUTION
# ======================================

def evolve(z, frame):
    rot = np.exp(1j * theta)

    # recursive RG update
    z = r * rot * z
    # orientation reversing
    if det_sign < 0:
        z = np.conjugate(z)

    # normalize explosion
    z = z / (1 + 0.015 * np.abs(z))

    # nonlinear recursive perturbation
    z += 0.08 * np.sin(frame * 0.03 + np.abs(z))

    return z


# ======================================
# INITIAL CLOUD
# ======================================

np.random.seed(0)

points = (
    np.random.normal(size=POINTS)
    + 1j * np.random.normal(size=POINTS)
)


# ======================================
# FRAME GENERATION
# ======================================

for frame in range(FRAMES):

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")

    new_points = []

    pulse = 0.5 + 0.5 * np.cos(frame * theta)

    for z in points:

        z_new = evolve(z, frame)
        new_points.append(z_new)

        x = np.real(z_new)
        y = np.imag(z_new)

        brightness = pulse * (1 / (1 + np.abs(z_new)))

        ax.add_patch(
            Circle(
                (x, y),
                0.015,
                alpha=min(1.0, brightness),

            )
        )
    points = np.array(new_points)

    lim = 6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        f"RG Visual Engine | r={r} θ={theta_deg}° det={det_sign}",
        color="white"
    )

    plt.tight_layout()

    filename = f"{OUTPUT_DIR}/frame_{frame:04d}.png"

    plt.savefig(
        filename,
        dpi=160,
        facecolor="black"
    )

    plt.close()

    print(f"Generated {filename}")
print("All frames generated.")
