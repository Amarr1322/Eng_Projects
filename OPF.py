
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Input Bus Data
busdata = np.array([
    [1, 1, 1.05, 0, 0, 0, 80, 0, 0, 0],  # Slack bus
    [2, 2, 1.02, 0, 50, 30, 50, 0, -20, 60],  # Generator bus
    [3, 2, 1.02, 0, 60, 35, 60, 0, -20, 60],  # Generator bus
    [4, 3, 1.0, 0, 70, 40, 0, 0, 0, 0],  # Load bus
    [5, 3, 1.0, 0, 90, 50, 0, 0, 0, 0],  # Load bus
])

# Input Line Data
linedata = np.array([
    [1, 2, 0.02, 0.04, 0.01],
    [1, 3, 0.01, 0.03, 0.01],
    [2, 4, 0.05, 0.20, 0.02],
    [3, 5, 0.05, 0.25, 0.02],
    [2, 5, 0.02, 0.10, 0.01],
    [4, 5, 0.01, 0.03, 0.01],
])

# Generator Cost Coefficients
gencost = np.array([
    [10, 5, 0.002],
    [15, 6, 0.003],
    [20, 7, 0.004],
])

# Power Limits for Generators
P_limits = np.array([
    [10, 100],
    [10, 80],
    [10, 90],
])

# Calculate Y-Bus Matrix
def calculate_ybus(linedata, n_buses):
    Ybus = np.zeros((n_buses, n_buses), dtype=complex)
    for line in linedata:
        from_bus, to_bus, R, X, B = int(line[0]) - 1, int(line[1]) - 1, line[2], line[3], line[4]
        Z = R + 1j * X
        Y = 1 / Z
        Ybus[from_bus, from_bus] += Y + 1j * B
        Ybus[to_bus, to_bus] += Y + 1j * B
        Ybus[from_bus, to_bus] -= Y
        Ybus[to_bus, from_bus] -= Y
    return Ybus

# Power Flow Analysis (Newton-Raphson)
def power_flow(Ybus, busdata):
    max_iter = 100
    tol = 1e-6
    n_buses = len(busdata)
    P = busdata[:, 4]
    Q = busdata[:, 5]
    V = busdata[:, 2]
    delta = busdata[:, 3]
    
    for _ in range(max_iter):
        mismatches = np.zeros(n_buses * 2)
        for i in range(n_buses):
            Pi, Qi = 0, 0
            for j in range(n_buses):
                Pi += abs(V[i]) * abs(V[j]) * (Ybus[i, j].real * np.cos(delta[i] - delta[j]) +
                                               Ybus[i, j].imag * np.sin(delta[i] - delta[j]))
                Qi += abs(V[i]) * abs(V[j]) * (Ybus[i, j].real * np.sin(delta[i] - delta[j]) -
                                               Ybus[i, j].imag * np.cos(delta[i] - delta[j]))
            mismatches[i] = P[i] - Pi
            mismatches[i + n_buses] = Q[i] - Qi

        if np.max(np.abs(mismatches)) < tol:
            break

        # Jacobian and update logic goes here (simplified for brevity)
        # Solve for voltage corrections using Newton-Raphson method

    return V, delta

# Objective Function (Cost Function)
def objective(P):
    total_cost = 0
    for i, (a, b, c) in enumerate(gencost):
        total_cost += a + b * P[i] + c * P[i] ** 2
    return total_cost

# Equality Constraint (Power Balance)
def power_balance_with_losses(P):
    total_load = np.sum(busdata[:, 4])  # Total load
    total_gen = np.sum(P)  # Total generation
    losses = 0.02 * total_gen  # Simplified losses (2% of total generation)
    return total_gen - total_load - losses

# Inequality Constraints (Generator Limits)
constraints = []
for i, (Pmin, Pmax) in enumerate(P_limits):
    constraints.append({'type': 'ineq', 'fun': lambda P, i=i, Pmin=Pmin: P[i] - Pmin})
    constraints.append({'type': 'ineq', 'fun': lambda P, i=i, Pmax=Pmax: Pmax - P[i]})

constraints.append({'type': 'eq', 'fun': power_balance_with_losses})

# Initial Guess
P_initial = np.array([50, 50, 50])

# Solve Optimization
result = minimize(objective, P_initial, constraints=constraints, method='SLSQP')

# Visualization
if result.success:
    print("Optimization Successful!")
    print("Optimal Generator Outputs (P):", result.x)
    print("Minimum Total Cost: $", result.fun)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(result.x) + 1), result.x, color='skyblue', label='Generator Outputs')
    plt.axhline(np.sum(busdata[:, 4]), color='red', linestyle='--', label='Total Load')
    plt.title("Optimal Generator Dispatch")
    plt.xlabel("Generator Number")
    plt.ylabel("Power Output (MW)")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Optimization Failed.")