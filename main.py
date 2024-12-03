import numpy as np
import matplotlib.pyplot as plt
from vasicek_model import vasicek_model, simulate_two_factor_vasicek
from bond_pricing import price_zero_coupon_bond, price_zero_coupon_bond_closed_form, price_coupon_bond

# User inputs
R0 = float(input("\nEnter initial interest rate: "))
alpha = float(input("\nEnter mean reversion speed: "))
mu = float(input("\nEnter long-term mean: "))
sigma = float(input("\nEnter volatility: "))
T = float(input("\nEnter time to maturity: "))
num_steps = int(input("\nEnter number of time steps: "))
num_paths = int(input("\nEnter number of Monte Carlo paths: "))
rho = float(input("\nEnter correlation between Wiener processes (0 to 1): "))

# One-Factor Simulation
R_sim = vasicek_model(R0, alpha, mu, sigma, T, num_steps, num_paths)
zero_coupon_price = price_zero_coupon_bond(R_sim, T / num_steps)
zero_coupon_price_closed = price_zero_coupon_bond_closed_form(R0, alpha, mu, sigma, T)

# Two-Factor Simulation
R1_sim, R2_sim = simulate_two_factor_vasicek(R0, R0, alpha, alpha, mu, mu, sigma, sigma, rho, T, num_steps, num_paths)

# Output
print(f"Zero-coupon bond price (Monte Carlo): {zero_coupon_price:.4f}")
print(f"Zero-coupon bond price (Closed-form): {zero_coupon_price_closed:.4f}")

# Plotting Interest Rate Paths (for the first few paths)
plt.figure(figsize=(10, 6))
for i in range(5):  # Plot first 5 paths for illustration
    plt.plot(R_sim[:, i], label=f'Path {i+1}')
plt.title('Interest Rate Paths (One-Factor Vasicek Model)')
plt.xlabel('Time Steps')
plt.ylabel('Interest Rate')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the bond prices comparison (Monte Carlo vs Closed-form)
plt.figure(figsize=(10, 6))
plt.bar(['Monte Carlo', 'Closed-form'], [zero_coupon_price, zero_coupon_price_closed], color=['blue', 'green'])
plt.title('Zero-Coupon Bond Price Comparison')
plt.ylabel('Bond Price')
plt.grid(True)
plt.show()

# Yield Curve Plotting for the One-Factor Model
def yield_curve_modeling(alpha, sigma, mu=0.03, R0=0.05, max_maturity=10, num_steps=100, num_paths=1000):
    maturities = np.linspace(1, max_maturity, 20)
    yields = []
    for maturity in maturities:
        R = vasicek_model(R0, alpha, mu, sigma, maturity, num_steps, num_paths)
        zero_coupon_price = price_zero_coupon_bond(R, T / num_steps)
        yield_maturity = -np.log(zero_coupon_price) / maturity
        yields.append(yield_maturity)
    return maturities, np.array(yields)

maturities, yields = yield_curve_modeling(alpha, sigma, mu, R0, 10)

plt.figure(figsize=(10, 6))
plt.plot(maturities, yields, label=f'α = {alpha}, σ = {sigma}, μ = {mu}')
plt.title('Yield Curve Using the Vasicek Model')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield')
plt.legend()
plt.grid(True)
plt.show()

# Sensitivity Plotting for Volatility and Mean Reversion
alphas = [0.1, 0.3, 0.5, 0.7, 1.0]
sigmas = [0.01, 0.02, 0.03, 0.04, 0.05]
results = np.zeros((len(alphas), len(sigmas)))

for i, alpha in enumerate(alphas):
    for j, sigma in enumerate(sigmas):
        results[i, j] = price_zero_coupon_bond(vasicek_model(R0, alpha, mu, sigma, T, num_steps, num_paths), T / num_steps)

plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alphas):
    plt.plot(sigmas, results[i, :], label=f'α = {alpha}')
plt.xlabel('Volatility (σ)')
plt.ylabel('Estimated Bond Price')
plt.title('Bond Price Sensitivity to Volatility and Mean Reversion')
plt.legend()
plt.grid(True)
plt.show()

