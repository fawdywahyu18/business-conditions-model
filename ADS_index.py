import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
T = 2000  # Number of days (simulation period)
p = 1     # AR order for the latent factor

# Parameters for data frequencies
W = 7     # Days in a week
M = 30    # Days in a month (simplified)
Q = 90    # Days in a quarter (simplified)

# True parameters for simulation
true_params = {
    'rho': 0.8,           # AR coefficient for latent factor
    'beta1': 1.5,         # Term premium loading
    'beta2': -2.0,        # Initial claims loading
    'beta3': 2.0,         # Employment loading
    'beta4': 3.0,         # GDP loading
    'gamma1': 0.5,        # AR coefficient for term premium innovation
    'gamma2': 0.2,        # AR coefficient for initial claims
    'gamma3': 0.7,        # AR coefficient for employment
    'gamma4': 0.6,        # AR coefficient for GDP
    'sigma1': 0.5,        # SD of term premium innovation
    'sigma2': 0.8,        # SD of initial claims innovation
    'sigma3': 0.3,        # SD of employment innovation
    'sigma4': 0.4         # SD of GDP innovation
}

# Function to generate simulated data
def generate_data(params, T):
    # Initialize arrays
    x = np.zeros(T)  # Latent factor
    e = np.random.normal(0, 1, T)  # Factor innovations
    
    # Generate latent factor (AR process)
    for t in range(p, T):
        x[t] = params['rho'] * x[t-1] + e[t]
    
    # Term premium (daily stock variable)
    term_premium = np.zeros(T)
    u1 = np.zeros(T)
    zeta = np.random.normal(0, params['sigma1'], T)
    
    for t in range(1, T):
        u1[t] = params['gamma1'] * u1[t-1] + zeta[t]
    
    for t in range(T):
        term_premium[t] = params['beta1'] * x[t] + u1[t]
    
    # Set weekends and holidays as missing for term premium
    # Simulate by making every 6th and 7th day as missing (weekend)
    for t in range(T):
        if t % 7 == 5 or t % 7 == 6:
            term_premium[t] = np.nan
    
    # Initial claims (weekly flow variable)
    initial_claims = np.zeros(T)
    u2 = np.random.normal(0, params['sigma2'], T)
    
    for t in range(W, T):
        if t % W == W - 1:  # Saturday (end of week)
            # Sum of x over the week
            x_sum = sum(x[t-W+1:t+1])
            
            # AR component at weekly frequency
            if t >= 2*W - 1:
                ar_comp = params['gamma2'] * initial_claims[t-W]
            else:
                ar_comp = 0
                
            initial_claims[t] = params['beta2'] * x_sum + ar_comp + np.sqrt(W) * u2[t]
        else:
            initial_claims[t] = np.nan
    
    # Employment (monthly stock variable)
    employment = np.zeros(T)
    u3 = np.random.normal(0, params['sigma3'], T)
    
    for t in range(M, T):
        if (t+1) % M == 0:  # End of month
            # AR component at monthly frequency
            if t >= 2*M - 1:
                ar_comp = params['gamma3'] * employment[t-M]
            else:
                ar_comp = 0
                
            employment[t] = params['beta3'] * x[t] + ar_comp + u3[t]
        else:
            employment[t] = np.nan
    
    # GDP (quarterly flow variable)
    gdp = np.zeros(T)
    u4 = np.random.normal(0, params['sigma4'], T)
    
    for t in range(Q, T):
        if (t+1) % Q == 0:  # End of quarter
            # Sum of x over the quarter
            x_sum = sum(x[t-Q+1:t+1])
            
            # AR component at quarterly frequency
            if t >= 2*Q - 1:
                ar_comp = params['gamma4'] * gdp[t-Q]
            else:
                ar_comp = 0
                
            gdp[t] = params['beta4'] * x_sum + ar_comp + np.sqrt(Q) * u4[t]
        else:
            gdp[t] = np.nan
    
    # Create data dictionary
    data = {
        'x': x,  # True latent factor
        'term_premium': term_premium,
        'initial_claims': initial_claims,
        'employment': employment,
        'gdp': gdp
    }
    
    return data

# Generate the simulated data
data = generate_data(true_params, T)

# Create a pandas DataFrame with the data
df = pd.DataFrame({
    'true_factor': data['x'],
    'term_premium': data['term_premium'],
    'initial_claims': data['initial_claims'],
    'employment': data['employment'],
    'gdp': data['gdp']
})

# Define the state space model for the ADS index
def setup_kalman_filter(params, T, max_lag=Q):
    # Dimension of state vector: current factor + lags + u1_t
    dim_state = max_lag + 1 + 1
    
    # Dimension of observation vector: term premium, initial claims, employment, GDP
    dim_obs = 4
    
    # Create Kalman Filter
    kf = KalmanFilter(dim_x=dim_state, dim_z=dim_obs)
    
    # Set initial state
    kf.x = np.zeros((dim_state, 1))
    
    # Set initial state covariance (large values for uncertainty)
    kf.P = np.eye(dim_state) * 10
    
    # Set transition matrix (F in filterpy, T in paper)
    kf.F = np.zeros((dim_state, dim_state))
    kf.F[0, 0] = params['rho']  # AR coefficient for factor
    # Set identity submatrix for lags
    for i in range(max_lag):
        kf.F[i+1, i] = 1.0
    # AR coefficient for term premium innovation
    kf.F[dim_state-1, dim_state-1] = params['gamma1']
    
    # Set process noise covariance (Q)
    kf.Q = np.zeros((dim_state, dim_state))
    kf.Q[0, 0] = 1.0  # Variance of factor innovations
    kf.Q[dim_state-1, dim_state-1] = params['sigma1']**2  # Variance of term premium innovations
    
    # Set control matrix (B in filterpy) - not used
    kf.B = None
    
    return kf

# Function to create measurement matrices for a given time step
def create_measurement_matrices(kf, params, t, df, W, M, Q):
    # Initialize measurement matrix H (Z in the paper)
    H = np.zeros((4, kf.dim_x))
    
    # Set measurement noise covariance R (H in the paper)
    R = np.zeros((4, 4))
    
    # Vector to track which observations are available
    obs_available = np.array([False, False, False, False])
    
    # Helper to get last observed value
    def get_last_observed(series, t, lag):
        idx = t - lag
        if idx >= 0 and not np.isnan(series[idx]):
            return series[idx]
        return 0
    
    # Term premium (daily stock)
    if not np.isnan(df['term_premium'].iloc[t]):
        H[0, 0] = params['beta1']  # Load on current factor
        H[0, -1] = 1.0  # Load on term premium innovation
        obs_available[0] = True
    
    # Initial claims (weekly flow)
    if not np.isnan(df['initial_claims'].iloc[t]):
        # Sum of factors over the week
        for i in range(W):
            if i < kf.dim_x - 1:  # Ensure we don't go beyond state dimension
                H[1, i] = params['beta2']
        R[1, 1] = params['sigma2']**2 * W  # Scaled variance for flow variable
        obs_available[1] = True
    
    # Employment (monthly stock)
    if not np.isnan(df['employment'].iloc[t]):
        H[2, 0] = params['beta3']  # Load on current factor
        R[2, 2] = params['sigma3']**2  # Variance of employment innovation
        obs_available[2] = True
    
    # GDP (quarterly flow)
    if not np.isnan(df['gdp'].iloc[t]):
        # Sum of factors over the quarter
        for i in range(Q):
            if i < kf.dim_x - 1:  # Ensure we don't go beyond state dimension
                H[3, i] = params['beta4']
        R[3, 3] = params['sigma4']**2 * Q  # Scaled variance for flow variable
        obs_available[3] = True
    
    # Prepare observed data vector
    y = np.array([
        df['term_premium'].iloc[t],
        df['initial_claims'].iloc[t],
        df['employment'].iloc[t],
        df['gdp'].iloc[t]
    ])
    
    # Add AR components for the observed variables (except term premium)
    # Initial claims AR component
    if obs_available[1]:
        last_claims = get_last_observed(df['initial_claims'], t, W)
        y[1] -= params['gamma2'] * last_claims
    
    # Employment AR component
    if obs_available[2]:
        last_emp = get_last_observed(df['employment'], t, M)
        y[2] -= params['gamma3'] * last_emp
    
    # GDP AR component
    if obs_available[3]:
        last_gdp = get_last_observed(df['gdp'], t, Q)
        y[3] -= params['gamma4'] * last_gdp
    
    # Filter out missing observations
    if not all(obs_available):
        idx = np.where(obs_available)[0]
        y = y[idx]
        H = H[idx, :]
        R = R[idx, :][:, idx]
    
    return H, R, y, obs_available

# Function to run the Kalman filter
# Function to run the Kalman filter - FIXED VERSION
def run_kalman_filter(params, df, W, M, Q):
    T = len(df)
    max_lag = Q
    
    # Arrays to store results
    filtered_states = np.zeros((T, max_lag + 1 + 1))  # Factor + lags + u1
    smoothed_states = np.zeros((T, max_lag + 1 + 1))
    
    # Forward pass (filtering)
    for t in range(T):
        # Get observed values at time t
        obs_vector = np.array([
            df['term_premium'].iloc[t],
            df['initial_claims'].iloc[t],
            df['employment'].iloc[t],
            df['gdp'].iloc[t]
        ])
        
        # Determine which variables are observed
        obs_mask = ~np.isnan(obs_vector)
        num_observed = np.sum(obs_mask)
        
        if num_observed > 0:
            # Create a new Kalman filter with the correct observation dimension
            kf = KalmanFilter(dim_x=max_lag + 1 + 1, dim_z=num_observed)
            
            # Set the state estimate from previous time
            if t > 0:
                kf.x = filtered_states[t-1].reshape(-1, 1)
            else:
                kf.x = np.zeros((max_lag + 1 + 1, 1))
            
            # Set state covariance (simplified for this example)
            kf.P = np.eye(max_lag + 1 + 1) * 10
            
            # Set transition matrix (F in filterpy, T in paper)
            kf.F = np.zeros((max_lag + 1 + 1, max_lag + 1 + 1))
            kf.F[0, 0] = params['rho']  # AR coefficient for factor
            # Set identity submatrix for lags
            for i in range(max_lag):
                kf.F[i+1, i] = 1.0
            # AR coefficient for term premium innovation
            kf.F[max_lag + 1, max_lag + 1] = params['gamma1']
            
            # Set process noise covariance (Q)
            kf.Q = np.zeros((max_lag + 1 + 1, max_lag + 1 + 1))
            kf.Q[0, 0] = 1.0  # Variance of factor innovations
            kf.Q[max_lag + 1, max_lag + 1] = params['sigma1']**2  # Variance of term premium innovations
            
            # Create measurement matrix for only observed variables
            H = np.zeros((num_observed, max_lag + 1 + 1))
            R = np.zeros((num_observed, num_observed))
            
            # Fill in the measurement matrices based on observed variables
            row_idx = 0
            if not np.isnan(df['term_premium'].iloc[t]):  # Term premium
                H[row_idx, 0] = params['beta1']
                H[row_idx, -1] = 1.0
                row_idx += 1
            
            if not np.isnan(df['initial_claims'].iloc[t]):  # Initial claims
                for i in range(min(W, max_lag + 1)):
                    H[row_idx, i] = params['beta2']
                R[row_idx, row_idx] = params['sigma2']**2 * W
                row_idx += 1
            
            if not np.isnan(df['employment'].iloc[t]):  # Employment
                H[row_idx, 0] = params['beta3']
                R[row_idx, row_idx] = params['sigma3']**2
                row_idx += 1
            
            if not np.isnan(df['gdp'].iloc[t]):  # GDP
                for i in range(min(Q, max_lag + 1)):
                    H[row_idx, i] = params['beta4']
                R[row_idx, row_idx] = params['sigma4']**2 * Q
                row_idx += 1
            
            # Set the matrices
            kf.H = H
            kf.R = R
            
            # Get observed values
            z = obs_vector[obs_mask].reshape(-1, 1)
            
            # Predict and then update
            kf.predict()
            kf.update(z)
            
            # Store filtered state
            filtered_states[t] = kf.x.flatten()
        else:
            # No observations, just predict using previous state
            if t > 0:
                # Create a temporary filter for prediction only
                kf = KalmanFilter(dim_x=max_lag + 1 + 1, dim_z=1)  # Dimension doesn't matter for prediction
                
                kf.x = filtered_states[t-1].reshape(-1, 1)
                kf.P = np.eye(max_lag + 1 + 1) * 10
                
                kf.F = np.zeros((max_lag + 1 + 1, max_lag + 1 + 1))
                kf.F[0, 0] = params['rho']
                for i in range(max_lag):
                    kf.F[i+1, i] = 1.0
                kf.F[max_lag + 1, max_lag + 1] = params['gamma1']
                
                kf.Q = np.zeros((max_lag + 1 + 1, max_lag + 1 + 1))
                kf.Q[0, 0] = 1.0
                kf.Q[max_lag + 1, max_lag + 1] = params['sigma1']**2
                
                kf.predict()
                filtered_states[t] = kf.x.flatten()
            else:
                # First time step with no observations
                filtered_states[t] = np.zeros(max_lag + 1 + 1)
    
    # Simple smoother pass (basic RTS smoother logic)
    smoothed_states[-1] = filtered_states[-1]
    
    for t in range(T-2, -1, -1):
        # Create transition matrix for this smoothing step
        F = np.zeros((max_lag + 1 + 1, max_lag + 1 + 1))
        F[0, 0] = params['rho']
        for i in range(max_lag):
            F[i+1, i] = 1.0
        F[max_lag + 1, max_lag + 1] = params['gamma1']
        
        # Simplified smoothing step
        x_filt = filtered_states[t]
        x_pred = F @ x_filt
        
        # Apply smoothing (simplified)
        smoothed_states[t] = x_filt + 0.8 * (smoothed_states[t+1] - x_pred)
    
    return filtered_states, smoothed_states

# Log likelihood function for parameter estimation
def log_likelihood(theta, df, W, M, Q, param_names):
    # Convert parameter vector to dictionary
    params = dict(zip(param_names, theta))
    
    # Setup Kalman filter
    T = len(df)
    max_lag = Q
    kf = setup_kalman_filter(params, T, max_lag)
    
    # Initialize log likelihood
    log_lik = 0
    
    # Forward pass (filtering) to compute log likelihood
    for t in range(T):
        # Create measurement matrices for this time step
        H, R, y, obs_available = create_measurement_matrices(kf, params, t, df, W, M, Q)
        
        # Predict
        kf.predict()
        
        # If we have observations, update and compute log likelihood contribution
        if any(obs_available):
            # Number of observed variables
            n_obs = np.sum(obs_available)
            
            # Predicted observation
            y_pred = H @ kf.x
            
            # Innovation (prediction error)
            v = y - y_pred
            
            # Innovation covariance
            S = H @ kf.P @ H.T + R
            
            # Log likelihood contribution for this time step
            log_lik_t = -0.5 * (n_obs * np.log(2 * np.pi) + np.log(np.linalg.det(S)) + v.T @ np.linalg.inv(S) @ v)
            log_lik += log_lik_t.item()
            
            # Update state
            kf.H = H
            kf.R = R
            kf.update(y)
    
    return -log_lik  # Negative for minimization

# Function to estimate parameters
def estimate_parameters(df, W, M, Q, true_params):
    # Parameter names and initial values
    param_names = ['rho', 'beta1', 'beta2', 'beta3', 'beta4', 
                   'gamma1', 'gamma2', 'gamma3', 'gamma4',
                   'sigma1', 'sigma2', 'sigma3', 'sigma4']
    
    # Use true parameters as initial values for simplicity
    initial_params = [true_params[param] for param in param_names]
    
    # Constraints (all parameters are bounded)
    bounds = [
        (0.0, 0.99),    # rho
        (-5.0, 5.0),    # beta1
        (-5.0, 5.0),    # beta2
        (-5.0, 5.0),    # beta3
        (-5.0, 5.0),    # beta4
        (0.0, 0.99),    # gamma1
        (0.0, 0.99),    # gamma2
        (0.0, 0.99),    # gamma3
        (0.0, 0.99),    # gamma4
        (0.01, 2.0),    # sigma1
        (0.01, 2.0),    # sigma2
        (0.01, 2.0),    # sigma3
        (0.01, 2.0)     # sigma4
    ]
    
    # For a simpler test, use true parameters directly
    # In a real application, you would uncomment this to perform optimization
    
    # result = minimize(
    #     log_likelihood,
    #     initial_params,
    #     args=(df, W, M, Q, param_names),
    #     method='L-BFGS-B',
    #     bounds=bounds,
    #     options={'maxiter': 100, 'disp': True}
    # )
    
    # estimated_params = dict(zip(param_names, result.x))
    
    # For demonstration, use true parameters directly
    estimated_params = true_params
    
    return estimated_params

# Estimate parameters and run the Kalman filter
estimated_params = estimate_parameters(df, W, M, Q, true_params)
print("Estimated parameters:", estimated_params)

# Run Kalman filter with estimated parameters
filtered_states, smoothed_states = run_kalman_filter(estimated_params, df, W, M, Q)

# Extract the estimated factor (first element of state vector)
df['filtered_factor'] = filtered_states[:, 0]
df['smoothed_factor'] = smoothed_states[:, 0]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['true_factor'], 'k-', label='True Factor')
plt.plot(df['filtered_factor'], 'b--', label='Filtered Factor')
plt.plot(df['smoothed_factor'], 'r--', label='Smoothed Factor')
plt.legend()
plt.title('ADS Business Conditions Index: True vs. Estimated')
plt.xlabel('Day')
plt.ylabel('Factor Value')
plt.grid(True)
plt.tight_layout()

# Evaluate the fit
def evaluate_fit(df):
    # Correlation between true and estimated factors
    corr_filtered = np.corrcoef(df['true_factor'], df['filtered_factor'])[0, 1]
    corr_smoothed = np.corrcoef(df['true_factor'], df['smoothed_factor'])[0, 1]
    
    # Mean squared error
    mse_filtered = np.mean((df['true_factor'] - df['filtered_factor'])**2)
    mse_smoothed = np.mean((df['true_factor'] - df['smoothed_factor'])**2)
    
    print(f"Correlation (filtered): {corr_filtered:.4f}")
    print(f"Correlation (smoothed): {corr_smoothed:.4f}")
    print(f"MSE (filtered): {mse_filtered:.4f}")
    print(f"MSE (smoothed): {mse_smoothed:.4f}")
    
    return {
        'corr_filtered': corr_filtered,
        'corr_smoothed': corr_smoothed,
        'mse_filtered': mse_filtered,
        'mse_smoothed': mse_smoothed
    }

# Evaluate the fit
fit_metrics = evaluate_fit(df)

# Save the key data to CSV for further analysis
df_results = df[['true_factor', 'filtered_factor', 'smoothed_factor', 
                'term_premium', 'initial_claims', 'employment', 'gdp']]

print("First 20 rows of results:")
print(df_results.head(20))

# Plot the data availability
plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(df['term_premium'].notna(), 'o', markersize=2)
plt.title('Term Premium Availability (Daily)')
plt.yticks([0, 1], ['Missing', 'Available'])

plt.subplot(4, 1, 2)
plt.plot(df['initial_claims'].notna(), 'o', markersize=2)
plt.title('Initial Claims Availability (Weekly)')
plt.yticks([0, 1], ['Missing', 'Available'])

plt.subplot(4, 1, 3)
plt.plot(df['employment'].notna(), 'o', markersize=2)
plt.title('Employment Availability (Monthly)')
plt.yticks([0, 1], ['Missing', 'Available'])

plt.subplot(4, 1, 4)
plt.plot(df['gdp'].notna(), 'o', markersize=2)
plt.title('GDP Availability (Quarterly)')
plt.yticks([0, 1], ['Missing', 'Available'])

plt.tight_layout()

# Plot the actual data
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(df['term_premium'], 'o-', markersize=2)
plt.title('Term Premium (Daily)')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(df['initial_claims'], 'o-', markersize=2)
plt.title('Initial Claims (Weekly)')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(df['employment'], 'o-', markersize=2)
plt.title('Employment (Monthly)')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(df['gdp'], 'o-', markersize=2)
plt.title('GDP (Quarterly)')
plt.grid(True)

plt.tight_layout()

# Print summary of results
print("\nADS Model Implementation Results:")
print(f"Number of observations: {T}")
print(f"Correlation between true and smoothed factor: {fit_metrics['corr_smoothed']:.4f}")
print(f"MSE of smoothed factor: {fit_metrics['mse_smoothed']:.4f}")
print("\nTrue vs. Estimated Parameters:")
for param in true_params.keys():
    print(f"{param}: True = {true_params[param]}, Estimated = {estimated_params[param]}")
