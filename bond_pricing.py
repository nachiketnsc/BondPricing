import numpy as np

# Zero-Coupon Bond Pricing (Monte Carlo)
def price_zero_coupon_bond(R, dt):
    bond_prices = np.exp(-np.cumsum(R, axis=0) * dt, dtype=np.float64)[-1]
    return np.mean(bond_prices)

# Zero-Coupon Bond Pricing (Closed-Form Vasicek)
def price_zero_coupon_bond_closed_form(R0, alpha, mu, sigma, T):
    B = (1 - np.exp(-alpha * T)) / alpha
    A = np.exp((mu - sigma**2 / (2 * alpha**2)) * (B - T) - sigma**2 * B**2 / (4 * alpha))
    return A * np.exp(-B * R0)

# Coupon Bond Pricing (Monte Carlo)
def price_coupon_bond(R, coupon_rate, coupon_frequency, num_paths, dt):
    num_coupons = int(T / coupon_frequency)
    bond_prices = np.zeros(num_paths)
    for i in range(num_paths):
        bond_price = 0
        for t in range(1, num_coupons + 1):
            discount_factor = np.exp(-np.sum(R[t * num_steps // num_coupons, i]) * dt)
            bond_price += coupon_rate * discount_factor
        bond_price += np.exp(-np.sum(R[-1, i]) * dt)
        bond_prices[i] = bond_price
    return np.mean(bond_prices)

