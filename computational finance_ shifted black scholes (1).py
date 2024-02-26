#!/usr/bin/env python
# coding: utf-8

# In[1]:


"the following code was designed for the pricing of a european type of payoff by using the shifted black scholes model when we are dealing with negative interest rates this code is inspired by the materials found in the book Mathematical modeling and computation in finance by L.A. Grzelak and Oosterlee"
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Function for calculating Black-Scholes call/put option price with shifted price
def BS_Call_Put_Option_Price_Shifted(CP, S_0, K, sigma, tau, r, shift):
    K_new = K + shift
    S_0_new = S_0 + shift
    return BS_Call_Put_Option_Price(CP, S_0_new, K_new, sigma, tau, r)

# Function for calculating Black-Scholes call/put option price
def BS_Call_Put_Option_Price(CP, S_0, K, sigma, tau, r):
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if CP == 'call':
        option_price = np.exp(-r * tau) * (S_0 * st.norm.cdf(d1) - K * st.norm.cdf(d2))
    elif CP == 'put':
        option_price = np.exp(-r * tau) * (K * st.norm.cdf(-d2) - S_0 * st.norm.cdf(-d1))
    return option_price

# Function for simulating geometric Brownian motion
def geometric_brownian_motion(mu, sigma, S0, dt, T, paths):
    n_steps = int(T / dt)
    t = np.linspace(0., T, n_steps + 1)
    W = np.random.standard_normal(size=(paths, n_steps + 1)) * np.sqrt(dt)
    W[:, 0] = 0.
    S = np.zeros_like(W)
    S[:, 0] = S0
    for i in range(1, n_steps + 1):
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * W[:, i])
    return S, t

def main():
    # Given parameters from ICAP the option has a straddle premıum
    S_0 = -0.345  # Initial LIBOR rate
    K = 2.456  # Strike price
    sigma = 0.2  # Volatility
    T = 4  # Time to maturity
    r = 0  # Risk-free rate (assuming it's zero for LIBOR)
    paths = 5  # Number of paths for simulation
    dt = 1 / 252  # Time increment (daily)

    # Optimal shift parameter
    shift = -S_0 / 2

    # Simulate geometric Brownian motion
    paths_data, t = geometric_brownian_motion(0, sigma, S_0, dt, T, paths)

    # Array for the shift parameters (necessary for plotting)
    shift_range = np.linspace(0.01, abs(S_0) * 2, 100)

    # Option prices for different shift parameters
    option_prices = [BS_Call_Put_Option_Price_Shifted('call', S_0, K, sigma, T, r, s) for s in shift_range]

    # Plotting for the geometric Brownian motion
    plt.figure(figsize=(10, 6))
    plt.title('Geometric Brownian Motion - Stock Price Simulation')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    for i in range(paths):
        plt.plot(t, paths_data[i])
    plt.grid(True)
    plt.show()

    # Plotting option prices
    plt.plot(shift_range, option_prices)
    plt.xlabel('Shift Parameter')
    plt.ylabel('Option Price')
    plt.title('Caplet Option Price vs Shift Parameter')
    plt.grid(True)
    plt.show()

main()


# In[ ]:




