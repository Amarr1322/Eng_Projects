

# Simulating radar detection and doppler shift for real-world scenarios using Python

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the radar system
frequency = 77e9  # Radar frequency in Hz (77 GHz)
speed_of_light = 3e8  # Speed of light in m/s
max_distance = 150  # Max detection range in meters
num_targets = 50  # Number of simulated targets

# Generate random distances for targets
distances = np.random.uniform(0, max_distance, num_targets)

# Generate random velocities for targets (-30 m/s to 30 m/s)
velocities = np.random.uniform(-30, 30, num_targets)

# Calculate Doppler shifts for each target
doppler_shifts = (2 * velocities * frequency) / speed_of_light

# Determine danger status based on threshold
threshold_distance = 44.44  # Threshold distance in meters
danger_status = distances < threshold_distance

# Plot the results
plt.figure(figsize=(12, 6))

# Subplot 1: Distance vs Target Index
plt.subplot(1, 2, 1)
plt.bar(range(num_targets), distances, color=['red' if danger else 'green' for danger in danger_status])
plt.axhline(y=threshold_distance, color='blue', linestyle='--', label='Safety Threshold')
plt.xlabel('Target Index')
plt.ylabel('Distance (m)')
plt.title('Target Distances')
plt.legend()

# Subplot 2: Velocity vs Doppler Shift
plt.subplot(1, 2, 2)
plt.scatter(velocities, doppler_shifts, c='purple')
plt.axhline(y=0, color='gray', linestyle='--', label='Stationary')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Doppler Shift (Hz)')
plt.title('Doppler Shift vs Velocity')
plt.legend()

plt.tight_layout()
plt.show()

# Summary of targets in danger zone
danger_indices = np.where(danger_status)[0]
print(f"Number of targets in danger zone: {len(danger_indices)}")
print(f"Indices of dangerous targets: {danger_indices}")

