import numpy as np
from scipy.stats import norm

def GBM_paths(S0, sigma, t, r, mu, n_sims, n_steps):
    """Simulates stock paths as geometric Brownian Motions
    Inputs:
    S0 (float): Underlying stock price at time 0
    sigma (float): Yearly volatility
    t (float): Time to expiration (years)
    r (float): Risk-free interest rate
    mu (float): Drift of log-returns
    n_sims (int): Number of simulated paths
    n_steps (int): Number of steps in each simulated path, each step interval has length t/n_steps

    Return (np.array): Array of stock paths
    """

    dt = t / n_steps
    noise = np.random.normal(loc=0, scale=1, size=(n_sims, n_steps))
    log_returns = (mu + r - sigma ** 2 * (0.5)) * dt + sigma * np.sqrt(dt) * noise
    exponent = np.cumsum(log_returns, axis=1)
    paths = S0 * np.exp(exponent)
    paths_with_start = np.insert(paths, 0, S0, axis=1)

    return paths_with_start

def LS_algorithm_call(paths_with_start, K, Ct_model):

    n_sims = paths_with_start.shape[0]
    n_steps = paths_with_start.shape[1]

    Vt = np.zeros((n_sims, n_steps))
    Vt[:,n_steps-1] = np.maximum(paths_with_start[:,n_steps-1]-K, 0)
    for i in range(n_steps-1, 0, -1):
        Ct_model.fit(X = paths_with_start[:, i-1:i], y = Vt[:,i])
        # Ct_estimate[:, i-1] = Ct_model.predict(X = paths_with_start[:, i-1])
        Vt[:,i-1] = np.maximum(paths_with_start[:,i-1]-K, Ct_model.predict(X = paths_with_start[:, i-1:i]))

    return np.mean(Vt[:,0])


def bs_call(S0, K, sigma, t, r):
    '''
    Black-Scholes Call Option formula

    Inputs:
    S0 (float): Stock price at time 0
    K (float): Strike Price
    sigma: Yearly volatility
    t: Time to expiration (years)
    r: Risk-free Interest rate


    Return:
    Black-Scholes value of call option (float)
    '''

    d1 = (np.log(S0 / K) + (r + (0.5) * sigma ** 2) * t) / (sigma * np.sqrt(t))

    d2 = d1 - sigma * np.sqrt(t)

    call_value = S0 * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

    return call_value


import numpy as np
import pandas as pd
from arch import arch_model

import numpy as np
import pandas as pd
from arch import arch_model

import numpy as np
import pandas as pd


def simulate_hybrid_vol_paths(garch_results,
                              best_error_model,
                              initial_sigma,
                              t,
                              n_sims,
                              n_steps,
                              predicted_weather):  # This will be a DataFrame
    """
    Simulates future volatility paths using a two-stage GARCH + Weather Error model.
    ...
    """
    # --- 1. Extract GARCH Parameters and Validate Inputs ---
    params = garch_results.params
    omega, alpha, beta = params['omega'], params['alpha[1]'], params['beta[1]']

    if len(predicted_weather) != n_steps:
        raise ValueError(f"Length of predicted_weather ({len(predicted_weather)}) must equal n_steps ({n_steps}).")

    # --- 3. Initialize Simulation Variables ---
    dt = t / n_steps
    variances = np.zeros((n_sims, n_steps + 1))
    variances[:, 0] = (initial_sigma ** 2) * dt
    random_shocks = np.random.normal(0, 1, size=(n_sims, n_steps))

    for i in range(1, n_steps + 1):
        prev_variances = variances[:, i - 1]
        z = random_shocks[:, i - 1]

        # The shock is based on the pure GARCH process
        shocks_squared = prev_variances * (z ** 2)

        # Update the variance using only the GARCH equation
        variances[:, i] = omega + alpha * shocks_squared + beta * prev_variances


    weather_adjustments = best_error_model.predict(predicted_weather)

    # Create the final variance array by adding the adjustment to the GARCH paths.
    # Note: We are adding a (n_steps,) array to a (n_sims, n_steps) slice.
    # Broadcasting in NumPy correctly adds the daily adjustment to every simulation path.
    variances[:, 1:] = variances[:, 1:] + weather_adjustments

    # Ensure variance is not negative
    variances = np.maximum(0, variances)

    return variances, random_shocks

def GBM_paths_with_volatility_paths(S0, r, t, volatility_paths, random_shocks):
    """
    Simulates asset paths using pre-computed, time-varying volatility paths.

    Inputs:
    S0 (float): Starting asset price.
    r (float): Risk-free interest rate (annualized).
    t (float): Time horizon in years.
    volatility_paths (np.ndarray): Array of simulated DAILY VARIANCES from the first function.
    random_shocks (np.ndarray): The random shocks used to generate the volatility paths.

    Returns:
    np.ndarray: Array of simulated asset paths of shape (n_sims, n_steps + 1).
    """
    n_sims, n_steps_plus_1 = volatility_paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = t / n_steps

    # Initialize the asset paths array
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0

    # --- 2. Run the Price Simulation Loop ---
    for i in range(1, n_steps + 1):
        # Use the volatility from the START of the step (i-1)
        # and the random shock for this step's interval
        variances_dt = volatility_paths[:, i - 1]
        z = random_shocks[:, i - 1]

        # Calculate log returns using the standard risk-neutral formula
        log_returns = (r * dt) - 0.5 * variances_dt + np.sqrt(variances_dt) * z

        # Update the asset paths
        paths[:, i] = paths[:, i - 1] * np.exp(log_returns)

    return paths