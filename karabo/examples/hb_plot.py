#!/usr/bin/env python3
from mwa_hyperbeam import FEEBeam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys

beam = FEEBeam("mwa_full_embedded_element_pattern.h5")
freq = 181e6
delays = [0] * 16
amps = [1] * 16

az = np.deg2rad(np.linspace(0, 359, 360))
za = np.deg2rad(np.linspace(0, 90, 91))
AZ, ZA = np.meshgrid(az, za)

J = beam.calc_jones_array(AZ.ravel(), ZA.ravel(), freq, delays, amps, True).reshape(
    91 * 360, 4
)
# Also print a sanity check at zenith
J0 = beam.calc_jones_array(np.array([0.0]), np.array([0.0]), freq, delays, amps, True)
print("zenith |Jxx|^2 =", float(np.abs(J0[0, 0]) ** 2))
XX = J[:, 0].reshape(91, 360)
P = np.abs(XX) ** 2

vmin = np.maximum(P[P > 0].min() if np.any(P > 0) else 1e-10, 1e-10)
vmax = P.max() if np.isfinite(P.max()) and P.max() > vmin else vmin * 10

# Report how many pixels are below 1e-9
threshold = 1e-9
below = int((P < threshold).sum())
total = int(P.size)
print(f"pixels < {threshold:g}: {below}/{total} ({below * 100.0 / total:.2f}%)")

plt.figure(figsize=(8, 7))
plt.imshow(P, origin="lower", aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(label="Power (linear)")
plt.title("Hyperbeam power — 181 MHz — X")
out = f"beam_{sys.argv[-1]}_181MHz_x.png"
plt.savefig(out, dpi=150)
print("Saved", out)
