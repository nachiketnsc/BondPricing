import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

# defining parameters for the Vasicek model
R0 = 0.05       # Initial interest rate
alpha = 0.1     # Mean reversion speed
mu = 0.03       # Long-term mean
sigma = 0.02    # Volatility
T = 5           # Time to maturity (in years)
num_steps = 100 # Number of time steps
num_simulations = 1000 # Number of Monte Carlo paths
dt = T / num_steps # Time step

# Simulating the interest rate using Euler Maruyama method

def vasicek_model(R0, alpha, mu, sigma, T, num_steps, num_simulations):
    R = np.zeros((num_steps+1, num_simulations))
    R[0] = R0
    for i in range(1, num_steps+1):
        dW = np.random.normal(0, np.sqrt(dt), num_simulations)
        R[i] = R[i-1] + alpha * (mu - R[i-1]) * dt + sigma * dW # Recurrence relation for the Vasicek model
    return R

# Pricing zero coupon bond using Monte Carlo simulation

def monte_carlo_bond_pricing(alpha, sigma, mu=0.03, R0=0.05, T=5, num_steps=100, num_simulations=1000):
    R = vasicek_model(R0, alpha, mu, sigma, T, num_steps, num_simulations)
    zero_coupon_price = price_zero_coupon_bond(R)
    return zero_coupon_price

#Vasicek model equation: dR = alpha(mu-R)dt + sigma*dW

def price_zero_coupon_bond(R):
    bond_prices = np.exp(-np.cumsum(R, axis=0) * dt, dtype=np.float64)[-1]  # Discounted cash flows
    return np.mean(bond_prices)  # Average over all paths

# Pricing zero coupon bond using the closed form solution of the Vasicek model

def price_zero_coupon_bond_closed_form(R0, alpha, mu, sigma, T):
    B = (1 - np.exp(-alpha * T)) / alpha
    A = np.exp((mu - sigma**2 / (2 * alpha**2)) * (B - T) - sigma**2 * B**2 / (4 * alpha))
    return A * np.exp(-B * R0)

# Pricing coupon bond using Monte Carlo simulation
def price_coupon_bond(R, coupon_rate=0.05, coupon_frequency=1):
    num_coupons = int(T / coupon_frequency)  # Number of coupon payments
    bond_prices = np.zeros(num_simulations)  # Initialize bond prices array

    for i in range(num_simulations):
        bond_price = 0
        for t in range(1, num_coupons + 1):
            coupon_payment = coupon_rate * np.exp(-np.sum(R[t * num_steps // num_coupons, i]) * dt)
            bond_price += coupon_payment
        bond_price += np.exp(-np.sum(R[-1, i]) * dt)  # Add principal repayment at maturity
        bond_prices[i] = bond_price  # Assign the total bond price as a scalar

    return np.mean(bond_prices)


# # Simulate and price both zero-coupon and coupon bonds
R_sim = vasicek_model(R0, alpha, mu, sigma, T, num_steps, num_simulations)  # Simulate interest rate paths
zero_coupon_price = price_zero_coupon_bond(R_sim)
coupon_bond_price = price_coupon_bond(R_sim)
# Price the zero coupon bond using the closed form solution
zero_coupon_price_closed_form = price_zero_coupon_bond_closed_form(R0, alpha, mu, sigma, T)
# compare the two prices

print('Zero coupon bond price using Monte Carlo simulation:', zero_coupon_price)
print('Zero coupon bond price using closed form solution:', zero_coupon_price_closed_form)

# Sensitivity analysis

# This part contains the sensitivity analysis of the bond price to the volatility (sigma), mean reversion (alpha), and time to maturity (T) (yield curve analysis) .
# Sensitivity analysis for alpha and sigma
alphas = [0.1, 0.3, 0.5, 0.7, 1.0]
sigmas = [0.01, 0.02, 0.03, 0.04, 0.05]
results = np.zeros((len(alphas), len(sigmas)))

for i, alpha in enumerate(alphas):
    for j, sigma in enumerate(sigmas):
        results[i, j] = monte_carlo_bond_pricing(alpha, sigma)

# Plotting the sensitivity results
plt.figure(figsize=(10, 6))
for i, alpha in enumerate(alphas):
    plt.plot(sigmas, results[i, :], label=f'β = {alpha}')
plt.xlabel('Volatility (σ)')
plt.ylabel('Estimated Bond Price')
plt.title('Bond Price Sensitivity to Volatility and Mean Reversion')
plt.legend()
plt.grid(True)
plt.show()

# Sensitivity analysis for Long term Mean (mu) of the interest rate
mus = [0.01, 0.03, 0.05, 0.07, 0.09]

results_extended=np.zeros((len(mus),len(sigmas),len(alphas)))

for k,alpha in enumerate(alphas):
    for j,sigma in enumerate(sigmas):
        for i,mu in enumerate(mus):
            results_extended[i,j,k]=monte_carlo_bond_pricing(alpha,sigma,mu)


# Plotting the sensitivity results
plt.figure(figsize=(10, 6))

for i, mu in enumerate(mus):
    for j, sigma in enumerate(sigmas):
        plt.plot(alphas, results_extended[i, j, :], label=f'μ = {mu}, σ = {sigma}')
plt.xlabel('Mean Reversion (α)')
plt.ylabel('Estimated Bond Price')
plt.title('Bond Price Sensitivity to Mean Reversion and Long-term Mean')
plt.legend()
plt.grid(True)
plt.show()

#Yield curve modelling
# This part contains the sensitivity analysis of the bond price to the time to maturity (T) (yield curve analysis).

# Yield Curve Modeling
def yield_curve_modeling(alpha, sigma, mu=0.03, R0=0.05, max_maturity=10, num_steps=100, num_simulations=1000):
    # List of different maturities
    maturities = np.linspace(1, max_maturity, 20)
    yields = []

    # Simulate bond prices for different maturities
    for maturity in maturities:
        # Simulate the interest rate paths using the Vasicek model for this maturity
        R = vasicek_model(R0, alpha, mu, sigma, maturity, num_steps, num_simulations)
        
        # Price the zero-coupon bond for the given maturity
        zero_coupon_price = price_zero_coupon_bond(R)
        
        # Calculate the yield for the given maturity
        yield_maturity = -np.log(zero_coupon_price) / maturity
        yields.append(yield_maturity)
    
    return maturities, np.array(yields)

# Define parameters for Yield Curve Modeling
max_maturity = 10  # Maximum maturity to simulate (e.g., 10 years)

# Simulate the Yield Curve
maturities, yields = yield_curve_modeling(alpha, sigma, mu, R0, max_maturity)

# Plot the Yield Curve
plt.figure(figsize=(10, 6))
plt.plot(maturities, yields, label=f'α = {alpha}, σ = {sigma}, μ = {mu}')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield')
plt.title('Yield Curve Using the Vasicek Model')
plt.legend()
plt.grid(True)
plt.show()

# Two-factor Vasicek model

# Randomly generate correlation (rho) between 0 and 1
rho = np.random.uniform(0, 1)

# Defining the parameters for the two-factor Vasicek model
alpha1 = 0.1
alpha2 = 0.2
mu1 = 0.03
mu2 = 0.04
sigma1 = 0.02
sigma2 = 0.03

# Simulate the two-factor Vasicek model using Euler Maruyama method
def simulate_two_factor_vasicek(R01, R02, alpha1, alpha2, mu1, mu2, sigma1, sigma2, rho, T, num_steps, num_simulations):
    dt = T / num_steps  # Time step size
    R1 = np.zeros((num_steps + 1, num_simulations))
    R2 = np.zeros((num_steps + 1, num_simulations))
    R1[0] = R01
    R2[0] = R02

    # Generate correlated Wiener processes
    for i in range(1, num_steps + 1):
        dW1 = np.random.normal(0, np.sqrt(dt), num_simulations)
        dW2 = rho * dW1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), num_simulations)

        # Euler-Maruyama method for two-factor Vasicek model
        R1[i] = R1[i-1] + alpha1 * (mu1 - R1[i-1]) * dt + sigma1 * dW1
        R2[i] = R2[i-1] + alpha2 * (mu2 - R2[i-1]) * dt + sigma2 * dW2

    return R1, R2

# Parameters for simulation
R01 = 0.03
R02 = 0.04
T = 1
num_steps = 100
num_simulations = 1000

# Ask user the input for the correlation parameter between the two wiener processes

rho = float(input("Enter the correlation parameter between the two Wiener processes (between 0 and 1): "))

# Simulating the short rates using the two-factor Vasicek model
R1_sim, R2_sim = simulate_two_factor_vasicek(R01, R02, alpha1, alpha2, mu1, mu2, sigma1, sigma2, rho, T, num_steps, num_simulations)

# Now, you can use R1_sim and R2_sim for further analysis, like bond pricing, etc.

def two_factor_zero_coupon_bond_pricing(R1, R2):
    bond_prices = np.exp(-np.cumsum(R1, axis=0) * dt, dtype=np.float64)[-1]  # Discounted cash flows
    return np.mean(bond_prices)  # Average over all paths

def two_factor_bond_pricing(R1, R2, coupon_rate=0.05, coupon_frequency=1):
    num_coupons = int(T / coupon_frequency)  # Number of coupon payments
    bond_prices = np.zeros(num_simulations)  # Initialize bond prices array

    for i in range(num_simulations):
        bond_price = 0
        for t in range(1, num_coupons + 1):
            discount_factor = np.exp(-np.sum((R1[t * num_steps // num_coupons, i] + R2[t * num_steps // num_coupons, i]) * dt))
            coupon_payment = coupon_rate * discount_factor
            bond_price += coupon_payment
        
        # Add principal repayment at maturity
        discount_factor_final = np.exp(-np.sum((R1[-1, i] + R2[-1, i]) * dt))
        bond_price += discount_factor_final
        
        bond_prices[i] = bond_price  # Assign the total bond price for the path

    return np.mean(bond_prices)  # Return the average bond price

# Price the zero coupon bond using the two-factor model

zero_coupon_price_two_factor = two_factor_zero_coupon_bond_pricing(R1_sim, R2_sim)

# Price the coupon bond using the two-factor model

coupon_bond_price_two_factor = two_factor_bond_pricing(R1_sim, R2_sim)

print('Zero coupon bond price using the two-factor Vasicek model:', zero_coupon_price_two_factor)
print('Coupon bond price using the two-factor Vasicek model:', coupon_bond_price_two_factor)
#end of code

