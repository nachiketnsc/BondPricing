import numpy as np

# One-Factor Vasicek Model Simulation
def vasicek_model(R0, alpha, mu, sigma, T, num_steps, num_paths):
    dt = T / num_steps
    R = np.zeros((num_steps + 1, num_paths))
    R[0] = R0
    for i in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), num_paths)
        R[i] = R[i-1] + alpha * (mu - R[i-1]) * dt + sigma * dW
    return R

# Two-Factor Vasicek Model Simulation
def simulate_two_factor_vasicek(R01, R02, alpha1, alpha2, mu1, mu2, sigma1, sigma2, rho, T, num_steps, num_simulations):
    dt = T / num_steps
    R1 = np.zeros((num_steps + 1, num_simulations))
    R2 = np.zeros((num_steps + 1, num_simulations))
    R1[0] = R01
    R2[0] = R02
    for i in range(1, num_steps + 1):
        dW1 = np.random.normal(0, np.sqrt(dt), num_simulations)
        dW2 = rho * dW1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), num_simulations)
        R1[i] = R1[i-1] + alpha1 * (mu1 - R1[i-1]) * dt + sigma1 * dW1
        R2[i] = R2[i-1] + alpha2 * (mu2 - R2[i-1]) * dt + sigma2 * dW2
    return R1, R2

