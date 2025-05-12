import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm, trange
from numba import jit, njit, prange

def create_date_sequence(start_date, end_date):
    """Generate a daily date sequence between start and end dates."""
    delta = end_date - start_date
    return [start_date + datetime.timedelta(days=i) for i in range(delta.days + 1)]

def count_days_in_period(date, freq):
    """Count number of days in the period (month/quarter/week) containing the date."""
    if freq == 'M':  # Monthly
        next_month = datetime.date(date.year + (date.month == 12), 
                                  (date.month % 12) + 1, 1)
        return (next_month - datetime.date(date.year, date.month, 1)).days
    elif freq == 'Q':  # Quarterly
        quarter_start_month = 3 * ((date.month - 1) // 3) + 1
        quarter_start = datetime.date(date.year, quarter_start_month, 1)
        next_quarter_year = date.year + (quarter_start_month == 10)
        next_quarter_month = (quarter_start_month + 3 - 1) % 12 + 1
        next_quarter = datetime.date(next_quarter_year, next_quarter_month, 1)
        return (next_quarter - quarter_start).days
    elif freq == 'W':  # Weekly
        return 7
    else:  # Daily
        return 1

def last_day_of_month(date):
    """Get the last day of the month for a given date."""
    if date.month == 12:
        return 31
    next_month = datetime.date(date.year, date.month + 1, 1)
    return (next_month - datetime.timedelta(days=1)).day

def last_day_of_quarter(date):
    """Get the last day of the quarter for a given date."""
    quarter = (date.month - 1) // 3
    if quarter == 0:  # Q1
        return last_day_of_month(datetime.date(date.year, 3, 1))
    elif quarter == 1:  # Q2
        return last_day_of_month(datetime.date(date.year, 6, 1))
    elif quarter == 2:  # Q3
        return last_day_of_month(datetime.date(date.year, 9, 1))
    else:  # Q4
        return 31  # December 31

def generate_simulated_data(start_date, end_date, seed=42):
    """
    Generate simulated data with typical characteristics for each variable type.
    
    Parameters:
    start_date: Start date for the simulation
    end_date: End date for the simulation
    seed: Random seed for reproducibility
    
    Returns:
    data_dict: Dictionary with simulated data
    true_factor: The true latent business conditions factor
    """
    np.random.seed(seed)
    
    # Create daily date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    print("Generating latent factor...")
    # Generate the latent business conditions factor (AR(2) process)
    ar_params = np.array([0.8, -0.2])  # AR(2) with moderate persistence
    true_factor = np.zeros(n_days)
    e_t = np.random.normal(0, 1, n_days)  # Innovation term
    
    # Initialize with stationary values
    true_factor[0] = e_t[0] / np.sqrt(1 - np.sum(ar_params))
    true_factor[1] = ar_params[0] * true_factor[0] + e_t[1]
    
    # Generate AR(2) process
    for t in trange(2, n_days, desc="Generating AR(2) process"):
        true_factor[t] = ar_params[0] * true_factor[t-1] + ar_params[1] * true_factor[t-2] + e_t[t]
    
    # Add a business cycle component (about 5-7 years)
    cycle_length = 365 * 6  # 6-year cycle
    cycle = 0.5 * np.sin(2 * np.pi * np.arange(n_days) / cycle_length)
    true_factor += cycle
    
    # Add a growth trend
    growth_trend = 0.0005 * np.arange(n_days)
    true_factor += growth_trend
    
    # Create DataFrame with the true factor
    factor_df = pd.DataFrame({'true_factor': true_factor}, index=date_range)
    
    print("Generating quarterly GDP data...")
    # Generate quarterly GDP data (flow variable)
    quarterly_dates = pd.date_range(start=start_date, end=end_date, freq='Q')
    gdp_base = np.zeros(len(quarterly_dates))
    
    # Map daily factor to quarterly frequency (averaging)
    for i, qdate in enumerate(tqdm(quarterly_dates, desc="Processing GDP data")):
        quarter_start = qdate - pd.DateOffset(months=2)
        quarter_days = factor_df[quarter_start:qdate].index
        # Sum the daily factors over the quarter (since GDP is a flow)
        gdp_factor_component = factor_df.loc[quarter_days, 'true_factor'].sum()
        gdp_base[i] = gdp_factor_component
    
    # Add GDP-specific components
    gdp_trend = 1000 + np.cumsum(np.random.normal(3, 0.5, len(quarterly_dates)))  # Strong upward trend
    gdp_seasonality = 20 * np.sin(2 * np.pi * np.arange(len(quarterly_dates)) / 4)  # Quarterly seasonality
    gdp_noise = np.random.normal(0, 30, len(quarterly_dates))  # Random noise
    
    # Combine components
    gdp_values = 50 * gdp_base + gdp_trend + gdp_seasonality + gdp_noise
    gdp_series = pd.Series(gdp_values, index=quarterly_dates)
    
    print("Generating monthly employment data...")
    # Generate monthly employment data (stock variable)
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='M')
    employment_base = np.zeros(len(monthly_dates))
    
    # Map daily factor to monthly frequency with a 3-month lag
    for i, mdate in enumerate(tqdm(monthly_dates, desc="Processing employment data")):
        # Employment lags by about 3 months
        if i >= 3:
            lagged_date = monthly_dates[i-3]
            employment_factor_component = factor_df.loc[lagged_date, 'true_factor']
            employment_base[i] = employment_factor_component
        else:
            # For the first 3 months, just use initial values
            employment_base[i] = factor_df.loc[mdate, 'true_factor']
    
    # Add employment-specific components
    employment_trend = 95000 + np.cumsum(np.random.normal(200, 50, len(monthly_dates)))  # Steady growth
    employment_seasonality = 1000 * np.sin(2 * np.pi * np.arange(len(monthly_dates)) / 12)  # Annual seasonality
    employment_noise = np.random.normal(0, 500, len(monthly_dates))  # Random noise
    
    # Combine components
    employment_values = 5000 * employment_base + employment_trend + employment_seasonality + employment_noise
    employment_series = pd.Series(employment_values, index=monthly_dates)
    
    print("Generating weekly jobless claims data...")
    # Generate weekly initial jobless claims (flow variable)
    weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    claims_base = np.zeros(len(weekly_dates))
    
    # Map daily factor to weekly frequency (negative correlation with business conditions)
    for i, wdate in enumerate(tqdm(weekly_dates, desc="Processing jobless claims data")):
        week_start = wdate - pd.DateOffset(days=6)
        week_days = factor_df[week_start:wdate].index
        # Sum the negative of daily factors over the week (since claims rise when conditions worsen)
        claims_factor_component = -factor_df.loc[week_days, 'true_factor'].sum()
        claims_base[i] = claims_factor_component
    
    # Add claims-specific components
    claims_trend = 300000 - np.cumsum(np.random.normal(100, 50, len(weekly_dates)))  # General downward trend
    claims_seasonality = 10000 * np.sin(2 * np.pi * np.arange(len(weekly_dates)) / 52)  # Annual seasonality
    claims_noise = np.random.normal(0, 20000, len(weekly_dates))  # High volatility
    
    # Combine components and ensure positive values
    claims_values = 5000 * claims_base + claims_trend + claims_seasonality + claims_noise
    claims_values = np.maximum(claims_values, 100000)  # Ensure positive values with a minimum
    claims_series = pd.Series(claims_values, index=weekly_dates)
    
    print("Generating daily yield curve data...")
    # Generate daily yield curve term premium (stock variable)
    yield_base = -0.2 * true_factor  # Negative correlation with business conditions
    
    # Add yield-specific components
    yield_trend = 2.5 + 0.0001 * np.arange(n_days)  # Slight upward trend
    yield_noise = np.random.normal(0, 0.05, n_days)  # Daily fluctuations
    
    # Combine components
    yield_values = yield_base + yield_trend + yield_noise
    yield_series = pd.Series(yield_values, index=date_range)
    
    # Create data dictionary
    data_dict = {
        'GDP': (gdp_series, 'Q', 'flow'),
        'Employment': (employment_series, 'M', 'stock'),
        'Jobless_Claims': (claims_series, 'W', 'flow'),
        'Yield_Curve': (yield_series, 'D', 'stock')
    }
    
    return data_dict, true_factor, factor_df

def prepare_data(data_dict, start_date, end_date):
    """
    Prepare the data for the state-space model.
    
    Parameters:
    data_dict: Dictionary with keys as variable names and values as tuples of 
               (pandas Series with data, frequency, variable_type)
    start_date: datetime.date object for the start of the analysis
    end_date: datetime.date object for the end of the analysis
    
    Returns:
    y_obs: Array of observations with NAs for missing data
    data_map: Information about data structure for the model
    """
    # Convert datetime.date to pandas timestamp if needed
    if isinstance(start_date, datetime.date):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, datetime.date):
        end_date = pd.Timestamp(end_date)
    
    # Create a dataframe with daily dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    n_vars = len(data_dict)
    
    # Dictionary to store mapping information
    data_map = {
        'variables': list(data_dict.keys()),
        'frequencies': {},
        'types': {},
        'Di': {},
        'obs_indices': {},
        'dates': date_range,
        'max_Di': 0  # Track maximum Di for state vector size
    }
    
    y_obs = np.full((n_days, n_vars), np.nan)
    
    for i, (var_name, (series, freq, var_type)) in enumerate(data_dict.items()):
        data_map['frequencies'][var_name] = freq
        data_map['types'][var_name] = var_type
        
        # Align data with our daily dates
        obs_indices = []
        for j, date in enumerate(tqdm(date_range, desc=f"Processing {var_name} data")):
            if date in series.index:
                y_obs[j, i] = series.loc[date]
                obs_indices.append(j)
        
        data_map['obs_indices'][var_name] = obs_indices
        
        # Calculate Di (days per period) for each observation
        if freq == 'D':
            data_map['Di'][var_name] = 1
            # Update max_Di if needed
            data_map['max_Di'] = max(data_map['max_Di'], 1)
        else:
            Di_values = {}
            for j in obs_indices:
                date = date_range[j]
                # Convert pandas timestamp to datetime.date for count_days_in_period
                if isinstance(date, pd.Timestamp):
                    date = date.date()
                days_in_period = count_days_in_period(date, freq)
                Di_values[j] = days_in_period
                # Update max_Di if needed
                data_map['max_Di'] = max(data_map['max_Di'], days_in_period)
            data_map['Di'][var_name] = Di_values
    
    return y_obs, data_map

@njit
def init_state_space_matrices(n_states, p, m, trend_order):
    """Initialize state space matrices with numba for speed."""
    # Initialize state transition matrix (T)
    T = np.zeros((n_states, n_states))
    
    # Set up shift matrix for lags
    if m > 1:
        # Create identity matrix for lags
        for i in range(1, m):
            T[i, i-1] = 1.0
    
    # Set up trend components (they remain constant over time)
    if trend_order > 0:
        for i in range(m, n_states):
            T[i, i] = 1.0
    
    # Selection matrix (R) for state innovations
    R = np.zeros((n_states, 1))
    R[0, 0] = 1.0  # Only the first element of state receives innovation
    
    # Innovation covariance matrix (Q)
    Q = np.ones((1, 1))  # Unit variance for identification
    
    # Initial state mean and covariance
    initial_state = np.zeros(n_states)
    initial_state_cov = np.eye(n_states) * 10.0
    
    return T, R, Q, initial_state, initial_state_cov

def construct_state_space_model(y_obs, data_map, p=2, trend_order=1):
    """
    Construct the state-space model matrices according to equations (9)-(11).
    
    Parameters:
    y_obs: Observation array with NAs for missing values
    data_map: Data mapping information
    p: AR order for the latent factor
    trend_order: Order of the trend polynomial
    
    Returns:
    ssm: Dictionary containing state-space matrices
    """
    n_days, n_vars = y_obs.shape
    
    # Determine state dimension based on AR order and flow variables
    max_Di = data_map['max_Di']
    m = max(p, max_Di)  # State dimension for AR process and flow variables
    
    # Total state dimension including trend components
    n_states = m + trend_order + 1  # +1 for constant term
    
    # Initialize matrices with numba acceleration
    T, R, Q, initial_state, initial_state_cov = init_state_space_matrices(
        n_states, p, m, trend_order)
    
    # Store dimensions and other information
    ssm = {
        'T': T,  # State transition matrix
        'R': R,  # Selection matrix
        'Q': Q,  # State innovation covariance
        'initial_state': initial_state,
        'initial_state_cov': initial_state_cov,
        'n_states': n_states,
        'n_obs': n_vars,
        'p': p,
        'trend_order': trend_order,
        'm': m  # Dimension of the dynamic part of the state
    }
    
    return ssm

@njit
def create_system_matrices_numba(n_vars, n_states, m, p, trend_order, 
                               obs_indices, var_types, Di_values, betas, sigmas, t):
    """Numba-optimized version of system matrices creation."""
    # Initialize matrices
    Z_t = np.zeros((n_vars, n_states))
    Gamma_t = np.zeros((n_vars, 1 + trend_order))
    H_t = np.diag(sigmas**2)
    w_t = np.ones(1 + trend_order)
    obs_vector = np.zeros(n_vars, dtype=np.bool_)
    
    # Populate trend terms
    for j in range(1, trend_order + 1):
        w_t[j] = t**j
    
    # Populate matrices for each variable
    for i in range(n_vars):
        if i in obs_indices:
            obs_vector[i] = True
            
            # Add constant term
            Gamma_t[i, 0] = 1.0
            
            # Add trend components
            for j in range(1, trend_order + 1):
                Gamma_t[i, j] = 1.0
            
            if var_types[i] == 0:  # Stock variable (0)
                # For stock variables: current value of the factor
                Z_t[i, 0] = betas[i]
                
            elif var_types[i] == 1:  # Flow variable (1)
                # For flow variables: sum over the period
                Di = Di_values[i]
                
                # Sum of the factor over the period (current and Di-1 lags)
                for j in range(min(Di, n_states)):
                    Z_t[i, j] = betas[i]
                
                # Adjust the variance for flow variables
                H_t[i, i] = sigmas[i]**2 * Di
    
    return Z_t, Gamma_t, H_t, w_t, obs_vector

def create_system_matrices(ssm, data_map, t, params):
    """
    Create the time-varying system matrices Z_t, Gamma_t, and H_t for time t.
    
    Parameters:
    ssm: State-space model structure
    data_map: Data mapping information
    t: Time index
    params: Model parameters
    
    Returns:
    Z_t: Observation matrix for time t
    Gamma_t: Coefficient matrix for predetermined variables
    H_t: Observation error covariance matrix
    w_t: Vector of predetermined variables (constant, trend, etc.)
    obs_vector: Vector indicating which variables are observed at time t
    """
    n_vars = ssm['n_obs']
    n_states = ssm['n_states']
    m = ssm['m']
    p = ssm['p']
    trend_order = ssm['trend_order']
    
    # Extract parameters
    betas = params['betas']
    sigmas = params['sigmas']
    
    # Prepare data for numba-optimized function
    obs_indices = []
    var_types = np.zeros(n_vars, dtype=np.int64)
    Di_values = np.ones(n_vars, dtype=np.int64)
    
    # Collect which variables are observed at time t
    for i, var_name in enumerate(data_map['variables']):
        if t in data_map['obs_indices'][var_name]:
            obs_indices.append(i)
            # Set variable type (0 for stock, 1 for flow)
            var_types[i] = 0 if data_map['types'][var_name] == 'stock' else 1
            # Get Di value
            if data_map['frequencies'][var_name] != 'D':
                Di = data_map['Di'][var_name][t] if isinstance(data_map['Di'][var_name], dict) else data_map['Di'][var_name]
                Di_values[i] = Di
    
    # Convert to numba-compatible types
    obs_indices_array = np.array(obs_indices, dtype=np.int64)
    
    # Call numba-optimized function
    Z_t, Gamma_t, H_t, w_t, obs_vector = create_system_matrices_numba(
        n_vars, n_states, m, p, trend_order, 
        obs_indices_array, var_types, Di_values, betas, sigmas, t)
    
    return Z_t, Gamma_t, H_t, w_t, obs_vector

@njit
def transform_system_numba(y_t, Z_t, Gamma_t, H_t, obs_vector):
    """Numba-optimized transformation for missing data."""
    # Count observed variables
    n_observed = np.sum(obs_vector)
    
    # If no variables are observed, return None for all
    if n_observed == 0:
        return None, None, None, None, None
    
    # Find indices of observed variables
    obs_indices = np.where(obs_vector)[0]
    
    # Extract observed components
    y_t_star = y_t[obs_indices]
    Z_t_star = Z_t[obs_indices, :]
    Gamma_t_star = Gamma_t[obs_indices, :]
    
    # Compute transformed error covariance
    H_t_star = np.zeros((n_observed, n_observed))
    for i in range(n_observed):
        for j in range(n_observed):
            H_t_star[i, j] = H_t[obs_indices[i], obs_indices[j]]
    
    # Create transformation matrix (for compatibility with non-numba code)
    W_t = np.zeros((n_observed, len(obs_vector)))
    for i in range(n_observed):
        W_t[i, obs_indices[i]] = 1.0
    
    return y_t_star, Z_t_star, Gamma_t_star, H_t_star, W_t

def transform_system_for_missing_data(y_t, Z_t, Gamma_t, H_t, w_t, obs_vector):
    """
    Transform the system to handle missing data according to equations (20)-(21).
    
    Parameters:
    y_t: Observation vector at time t (with NAs)
    Z_t: Observation matrix
    Gamma_t: Coefficient matrix for predetermined variables
    H_t: Observation error covariance
    w_t: Vector of predetermined variables
    obs_vector: Boolean vector indicating which variables are observed
    
    Returns:
    y_t_star: Observation vector with only observed variables
    Z_t_star: Transformed observation matrix
    Gamma_t_star: Transformed coefficient matrix
    H_t_star: Transformed error covariance matrix
    """
    return transform_system_numba(y_t, Z_t, Gamma_t, H_t, obs_vector)

@njit
def kalman_filter_step(a_t, P_t, y_t, Z_t, Gamma_t, H_t, T, R, Q, w_t, obs_vector):
    """Perform one step of the Kalman filter with numba acceleration."""
    # Store the predicted state and covariance
    predicted_state = a_t.copy()
    predicted_cov = P_t.copy()
    
    # Check if any variables are observed
    if np.any(obs_vector):
        # Transform system to handle missing data
        y_t_star, Z_t_star, Gamma_t_star, H_t_star, _ = transform_system_numba(
            y_t, Z_t, Gamma_t, H_t, obs_vector)
        
        # Compute innovation (v_t)
        v_t = y_t_star - Z_t_star @ a_t - Gamma_t_star @ w_t
        
        # Compute innovation covariance (F_t)
        F_t = Z_t_star @ P_t @ Z_t_star.T + H_t_star
        
        # Ensure numerical stability
        F_t = 0.5 * (F_t + F_t.T)  # Make sure F_t is symmetric
        
        # Try Cholesky decomposition to check if F_t is positive definite
        try:
            # Using modified Cholesky for stability
            L = np.zeros_like(F_t)
            n = F_t.shape[0]
            
            for i in range(n):
                L[i, i] = F_t[i, i]
                for j in range(i):
                    L[i, i] -= L[i, j]**2
                
                if L[i, i] <= 1e-8:  # Small positive number for numerical stability
                    # Not positive definite, use regularization
                    L[i, i] = 1e-8
                else:
                    L[i, i] = np.sqrt(L[i, i])
                
                for j in range(i+1, n):
                    L[j, i] = F_t[j, i]
                    for k in range(i):
                        L[j, i] -= L[j, k] * L[i, k]
                    L[j, i] /= L[i, i]
            
            # Compute log determinant
            log_det_F_t = 0.0
            for i in range(n):
                log_det_F_t += 2.0 * np.log(L[i, i])
            
            # Solve system for innovation
            # First solve L*y = v_t
            y = np.zeros_like(v_t)
            for i in range(n):
                y[i] = v_t[i]
                for j in range(i):
                    y[i] -= L[i, j] * y[j]
                y[i] /= L[i, i]
            
            # Then solve L.T*x = y
            inv_F_t_v_t = np.zeros_like(v_t)
            for i in range(n-1, -1, -1):
                inv_F_t_v_t[i] = y[i]
                for j in range(i+1, n):
                    inv_F_t_v_t[i] -= L[j, i] * inv_F_t_v_t[j]
                inv_F_t_v_t[i] /= L[i, i]
            
            # Log-likelihood contribution
            loglik_t = -0.5 * (n * np.log(2.0 * np.pi) + log_det_F_t + 
                             np.sum(v_t * inv_F_t_v_t))
            
            # Kalman gain
            K_t = P_t @ Z_t_star.T @ np.linalg.solve(F_t, np.eye(F_t.shape[0]))
            
            # Update filtered state and covariance
            a_t_t = a_t + K_t @ v_t
            P_t_t = P_t - K_t @ Z_t_star @ P_t
            
            # Ensure P_t_t is symmetric
            P_t_t = 0.5 * (P_t_t + P_t_t.T)
            
            valid_update = True
            
        except:
            # If numerical issues, skip the update
            a_t_t = a_t
            P_t_t = P_t
            loglik_t = 0.0
            valid_update = False
    else:
        # No observations at time t
        a_t_t = a_t
        P_t_t = P_t
        loglik_t = 0.0
        valid_update = False
    
    # Prediction step
    a_t_plus1 = T @ a_t_t
    P_t_plus1 = T @ P_t_t @ T.T + R @ Q @ R.T
    
    # Ensure P_t_plus1 is symmetric
    P_t_plus1 = 0.5 * (P_t_plus1 + P_t_plus1.T)
    
    return a_t_t, P_t_t, a_t_plus1, P_t_plus1, loglik_t, valid_update

def kalman_filter(y_obs, data_map, ssm, params):
   """
   Run the Kalman filter according to equations (12)-(19) with progress bar.
   
   Parameters:
   y_obs: Observation array
   data_map: Data mapping information
   ssm: State-space model
   params: Model parameters
   
   Returns:
   filtered_states: Filtered state estimates (a_t|t)
   filtered_covs: Filtered state covariances (P_t|t)
   predicted_states: Predicted state estimates (a_t)
   predicted_covs: Predicted state covariances (P_t)
   loglik: Log-likelihood value
   """
   n_days, n_vars = y_obs.shape
   n_states = ssm['n_states']
   
   # Initialize containers
   filtered_states = np.zeros((n_days, n_states))
   filtered_covs = np.zeros((n_days, n_states, n_states))
   predicted_states = np.zeros((n_days, n_states))
   predicted_covs = np.zeros((n_days, n_states, n_states))
   
   # For log-likelihood calculation
   loglik = 0.0
   
   # Initialize state
   a_t = ssm['initial_state'].copy()
   P_t = ssm['initial_state_cov'].copy()
   
   # Set up AR coefficients in transition matrix
   T = ssm['T'].copy()
   p = ssm['p']
   
   if p == 1:
       T[0, 0] = params['ar_coeffs'][0]
   else:
       T[0, :p] = params['ar_coeffs']
   
   # Kalman filter recursion with progress bar
   for t in tqdm(range(n_days), desc="Running Kalman filter"):
       # Create time-varying system matrices
       Z_t, Gamma_t, H_t, w_t, obs_vector = create_system_matrices(ssm, data_map, t, params)
       
       # Extract observation vector for time t
       y_t = y_obs[t]
       
       # Perform one step of the Kalman filter
       a_t_t, P_t_t, a_t_next, P_t_next, loglik_t, valid_update = kalman_filter_step(
           a_t, P_t, y_t, Z_t, Gamma_t, H_t, T, ssm['R'], ssm['Q'], w_t, obs_vector)
       
       # Store results
       predicted_states[t] = a_t
       predicted_covs[t] = P_t
       filtered_states[t] = a_t_t
       filtered_covs[t] = P_t_t
       
       # Update log-likelihood
       if valid_update:
           loglik += loglik_t
       
       # Update state for next iteration
       a_t = a_t_next
       P_t = P_t_next
   
   return filtered_states, filtered_covs, predicted_states, predicted_covs, loglik

@njit
def kalman_smoother_step(t, filtered_state_t, filtered_cov_t, predicted_state_tp1, 
                       predicted_cov_tp1, smoothed_state_tp1, smoothed_cov_tp1, T):
   """Perform one step of the Kalman smoother with numba acceleration."""
   # Compute smoothing gain (J_t)
   J_t = filtered_cov_t @ T.T @ np.linalg.inv(predicted_cov_tp1)
   
   # Smooth the state and covariance
   smoothed_state_t = filtered_state_t + J_t @ (smoothed_state_tp1 - predicted_state_tp1)
   smoothed_cov_t = filtered_cov_t + J_t @ (smoothed_cov_tp1 - predicted_cov_tp1) @ J_t.T
   
   # Ensure smoothed_cov is symmetric
   smoothed_cov_t = 0.5 * (smoothed_cov_t + smoothed_cov_t.T)
   
   return smoothed_state_t, smoothed_cov_t

def kalman_smoother(y_obs, data_map, ssm, params, filtered_states, filtered_covs, predicted_states, predicted_covs):
   """
   Run the Kalman smoother with progress bar.
   
   Parameters:
   y_obs: Observation array
   data_map: Data mapping information
   ssm: State-space model
   params: Model parameters
   filtered_states: Filtered state estimates (a_t|t)
   filtered_covs: Filtered state covariances (P_t|t)
   predicted_states: Predicted state estimates (a_t)
   predicted_covs: Predicted state covariances (P_t)
   
   Returns:
   smoothed_states: Smoothed state estimates
   smoothed_covs: Smoothed state covariances
   """
   n_days, n_vars = y_obs.shape
   n_states = ssm['n_states']
   
   # Initialize containers for smoothed estimates
   smoothed_states = np.zeros((n_days, n_states))
   smoothed_covs = np.zeros((n_days, n_states, n_states))
   
   # Initialize with the last filtered values
   smoothed_states[-1] = filtered_states[-1]
   smoothed_covs[-1] = filtered_covs[-1]
   
   # Set up AR coefficients in transition matrix
   T = ssm['T'].copy()
   p = ssm['p']
   
   if p == 1:
       T[0, 0] = params['ar_coeffs'][0]
   else:
       T[0, :p] = params['ar_coeffs']
   
   # Backward recursion with progress bar
   for t in tqdm(range(n_days - 2, -1, -1), desc="Running Kalman smoother"):
       # Perform one step of the Kalman smoother
       smoothed_states[t], smoothed_covs[t] = kalman_smoother_step(
           t, filtered_states[t], filtered_covs[t], 
           predicted_states[t+1], predicted_covs[t+1], 
           smoothed_states[t+1], smoothed_covs[t+1], T)
   
   return smoothed_states, smoothed_covs

def log_likelihood_function(params_vector, y_obs, data_map, ssm_template):
   """
   Log-likelihood function for parameter estimation using equation (22).
   
   Parameters:
   params_vector: Vector of parameters to estimate
   y_obs: Observation array
   data_map: Data mapping information
   ssm_template: Template for state-space model
   
   Returns:
   neg_loglik: Negative log-likelihood
   """
   n_vars = ssm_template['n_obs']
   p = ssm_template['p']
   
   # Unpack parameters
   i = 0
   ar_coeffs = params_vector[i:i+p]
   i += p
   
   betas = params_vector[i:i+n_vars]
   i += n_vars
   
   sigmas = params_vector[i:i+n_vars]
   
   # Create parameter dictionary
   params = {
       'ar_coeffs': ar_coeffs,
       'betas': betas,
       'sigmas': sigmas
   }
   
   # Create state-space model
   ssm = ssm_template.copy()
   
   # Set AR coefficients in transition matrix
   if p == 1:
       ssm['T'][0, 0] = ar_coeffs[0]
   else:
       ssm['T'][0, :p] = ar_coeffs
   
   # Run Kalman filter to compute log-likelihood
   try:
       _, _, _, _, loglik = kalman_filter(y_obs, data_map, ssm, params)
       return -loglik  # Return negative log-likelihood for minimization
   except Exception as e:
       # If filter fails, return a large value
       print(f"Warning: Kalman filter failed with error: {e}")
       return 1e10

def estimate_model(y_obs, data_map, p=2, trend_order=1, max_iter=1000):
   """
   Estimate the model parameters using maximum likelihood.
   
   Parameters:
   y_obs: Observation array
   data_map: Data mapping information
   p: AR order for the latent factor
   trend_order: Order of the trend polynomial
   max_iter: Maximum number of iterations for optimization
   
   Returns:
   params: Dictionary of estimated parameters
   ssm: Estimated state-space model
   """
   # Create template for state-space model
   ssm_template = construct_state_space_model(y_obs, data_map, p, trend_order)
   
   n_vars = len(data_map['variables'])
   
   # Initial parameter values
   initial_ar_coeffs = np.array([0.5, 0.3]) if p == 2 else np.array([0.7])  # Initial AR coefficients
   initial_betas = np.ones(n_vars)  # Initial factor loadings
   initial_sigmas = np.ones(n_vars)  # Initial innovation standard deviations
   
   # Combine initial parameters into a vector
   initial_params = np.concatenate([
       initial_ar_coeffs,
       initial_betas,
       initial_sigmas
   ])
   
   # Set bounds for optimization
   bounds = []
   
   # AR coefficient bounds (ensure stationarity)
   if p == 1:
       bounds.append((-0.99, 0.99))  # AR(1) coefficient
   else:
       # For AR(2), ensure stationarity
       bounds.extend([(-0.99, 0.99), (-0.99, 0.99)])
   
   # Beta (factor loading) bounds - first beta normalized to 1 for identification
   bounds.append((1.0, 1.0))  # Fix first beta to 1
   bounds.extend([(-10, 10)] * (n_vars - 1))  # Other betas are free
   
   # Sigma (innovation std) bounds
   bounds.extend([(0.01, 100)] * n_vars)
   
   # Progress callback for optimization
   iteration = [0]
   def callback(xk):
       iteration[0] += 1
       if iteration[0] % 10 == 0:  # Update every 10 iterations
           print(f"Optimization iteration {iteration[0]}/{max_iter}")
   
   # Perform optimization
   print("Starting parameter estimation...")
   result = optimize.minimize(
       log_likelihood_function,
       initial_params,
       args=(y_obs, data_map, ssm_template),
       method='L-BFGS-B',
       bounds=bounds,
       callback=callback,
       options={'maxiter': max_iter, 'disp': True}
   )
   
   if not result.success:
       print(f"Warning: Optimization did not converge: {result.message}")
   
   # Extract estimated parameters
   params_vector = result.x
   
   # Unpack parameters
   i = 0
   ar_coeffs = params_vector[i:i+p]
   i += p
   
   betas = params_vector[i:i+n_vars]
   i += n_vars
   
   sigmas = params_vector[i:i+n_vars]
   
   # Create parameter dictionary
   params = {
       'ar_coeffs': ar_coeffs,
       'betas': betas,
       'sigmas': sigmas
   }
   
   # Update state-space model with estimated parameters
   ssm = ssm_template.copy()
   if p == 1:
       ssm['T'][0, 0] = ar_coeffs[0]
   else:
       ssm['T'][0, :p] = ar_coeffs
   
   print("Estimation completed.")
   print(f"AR coefficients: {ar_coeffs}")
   print(f"Beta coefficients: {betas}")
   print(f"Sigma coefficients: {sigmas}")
   
   return params, ssm

@njit
def forecast_business_conditions_numba(T, R, Q, last_state, forecast_horizon):
   """Numba-accelerated forecasting function."""
   n_states = len(last_state)
   
   # Initialize forecast containers
   forecasts = np.zeros((forecast_horizon, n_states))
   forecast_covs = np.zeros((forecast_horizon, n_states, n_states))
   
   # Initial state for forecasting
   forecast_state = last_state.copy()
   forecast_cov = np.eye(n_states)  # Identity as starting covariance
   
   # Generate forecasts using the state transition equation
   for h in range(forecast_horizon):
       # State forecasting using transition equation
       forecast_state = T @ forecast_state
       forecast_cov = T @ forecast_cov @ T.T + R @ Q @ R.T
       
       # Ensure covariance is symmetric
       forecast_cov = 0.5 * (forecast_cov + forecast_cov.T)
       
       # Store results
       forecasts[h] = forecast_state
       forecast_covs[h] = forecast_cov
   
   return forecasts, forecast_covs

def forecast_business_conditions(ssm, params, filtered_states, forecast_horizon=30):
   """
   Forecast business conditions.
   
   Parameters:
   ssm: State-space model dictionary
   params: Dictionary of model parameters
   filtered_states: Filtered state estimates
   forecast_horizon: Number of days to forecast
   
   Returns:
   forecasts: Forecasted business conditions
   forecast_covs: Forecast covariances
   """
   # Set up AR coefficients in transition matrix
   T = ssm['T'].copy()
   p = ssm['p']
   
   if p == 1:
       T[0, 0] = params['ar_coeffs'][0]
   else:
       T[0, :p] = params['ar_coeffs']
   
   # Initial state is the last filtered state
   last_state = filtered_states[-1].copy()
   
   # Use numba-accelerated function for forecasting
   print(f"Forecasting {forecast_horizon} days ahead...")
   forecasts, forecast_covs = forecast_business_conditions_numba(
       T, ssm['R'], ssm['Q'], last_state, forecast_horizon)
   
   return forecasts, forecast_covs

def plot_business_conditions(filtered_states, smoothed_states, dates, true_factor=None, 
                          forecasts=None, forecast_dates=None, confidence_level=0.9):
   """
   Plot estimated and forecasted business conditions.
   
   Parameters:
   filtered_states: Filtered state estimates
   smoothed_states: Smoothed state estimates
   dates: Date indices
   true_factor: True factor values (optional, for simulated data)
   forecasts: Forecasted states (optional)
   forecast_dates: Dates for forecasts (optional)
   confidence_level: Confidence level for prediction intervals
   """
   plt.figure(figsize=(12, 8))
   
   # Extract the factor (first state variable)
   factor_filtered = filtered_states[:, 0]
   factor_smoothed = smoothed_states[:, 0]
   
   # Plot filtered and smoothed estimates
   plt.plot(dates, factor_filtered, 'b--', alpha=0.5, linewidth=1, label='Filtered')
   plt.plot(dates, factor_smoothed, 'r-', linewidth=2, label='Smoothed')
   
   # Plot true factor if provided (for simulated data)
   if true_factor is not None:
       plt.plot(dates, true_factor, 'k-', linewidth=1, label='True Factor')
   
   # Plot forecasts if provided
   if forecasts is not None and forecast_dates is not None:
       forecast_mean = forecasts[:, 0]
       plt.plot(forecast_dates, forecast_mean, 'g--', linewidth=2, label='Forecast')
       
       # Add confidence intervals for forecasts
       # (This is simplified - would need forecast_covs for proper intervals)
       z_value = 1.96  # Approximate 95% confidence interval
       forecast_std = np.sqrt(np.linspace(0.1, 1.0, len(forecast_mean)))  # Simplified increasing uncertainty
       plt.fill_between(forecast_dates, 
                        forecast_mean - z_value * forecast_std,
                        forecast_mean + z_value * forecast_std, 
                        color='g', alpha=0.2)
   
   # Add recession shading (if available)
   # This would be added here if recession dates were known
   
   plt.title('Business Conditions Index', fontsize=14)
   plt.xlabel('Date', fontsize=12)
   plt.ylabel('Index Value', fontsize=12)
   plt.legend(fontsize=10)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   
   # Add a second plot showing the comparison between filtered and smoothed
   plt.figure(figsize=(12, 4))
   plt.plot(dates, factor_smoothed - factor_filtered, 'b-', linewidth=1)
   plt.title('Difference Between Smoothed and Filtered Estimates', fontsize=14)
   plt.xlabel('Date', fontsize=12)
   plt.ylabel('Difference', fontsize=12)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   
   plt.show()

def plot_indicators_with_factor(y_obs, data_map, smoothed_states):
   """
   Plot each economic indicator alongside the estimated factor.
   
   Parameters:
   y_obs: Observation array
   data_map: Data mapping information
   smoothed_states: Smoothed state estimates
   """
   n_vars = len(data_map['variables'])
   factor = smoothed_states[:, 0]
   dates = data_map['dates']
   
   fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3*n_vars), sharex=True)
   
   # Normalize factor for plotting
   factor_norm = (factor - np.mean(factor)) / np.std(factor)
   
   for i, var_name in enumerate(data_map['variables']):
       ax = axes[i] if n_vars > 1 else axes
       
       # Get observed values and observation dates for this variable
       obs_indices = data_map['obs_indices'][var_name]
       obs_dates = [dates[j] for j in obs_indices]
       obs_values = y_obs[obs_indices, i]
       
       # Normalize observed values for comparison
       obs_norm = (obs_values - np.mean(obs_values)) / np.std(obs_values)
       
       # Plot normalized factor
       ax.plot(dates, factor_norm, 'r-', alpha=0.5, label='Business Conditions Factor (normalized)')
       
       # Plot normalized indicator
       ax.plot(obs_dates, obs_norm, 'b.-', label=f'{var_name} (normalized)')
       
       ax.set_title(f'{var_name} vs Business Conditions Factor', fontsize=12)
       ax.grid(True, alpha=0.3)
       ax.legend(fontsize=10)
   
   plt.tight_layout()
   plt.show()

def run_real_time_measurement(data_dict, start_date=None, end_date=None, p=2, trend_order=1, 
                           true_factor=None, forecast_horizon=30):
   """
   Run the real-time measurement of business conditions following the 
   state-space representation from equations (9)-(22).
   
   Parameters:
   data_dict: Dictionary with keys as variable names and values as tuples of 
             (pandas Series with data, frequency, variable_type)
   start_date: Start date for analysis (default: earliest date in data)
   end_date: End date for analysis (default: latest date in data)
   p: AR order for the latent factor
   trend_order: Order of the trend polynomial
   true_factor: True factor values (optional, for simulated data)
   forecast_horizon: Number of days to forecast
   
   Returns:
   results: Dictionary containing model results
   """
   # Determine date range if not provided
   if start_date is None or end_date is None:
       all_dates = []
       for series, _, _ in data_dict.values():
           all_dates.extend(series.index)
       all_dates = sorted(all_dates)
       
       if start_date is None:
           start_date = all_dates[0]
       if end_date is None:
           end_date = all_dates[-1]
   
   print(f"Analysis period: {start_date} to {end_date}")
   
   # Prepare the data
   print("Preparing data...")
   y_obs, data_map = prepare_data(data_dict, start_date, end_date)
   
   # Estimate the model
   print("Estimating model parameters...")
   params, ssm = estimate_model(y_obs, data_map, p, trend_order)
   
   # Run Kalman filter
   print("Running Kalman filter...")
   filtered_states, filtered_covs, predicted_states, predicted_covs, loglik = kalman_filter(
       y_obs, data_map, ssm, params)
   
   # Run Kalman smoother
   print("Running Kalman smoother...")
   smoothed_states, smoothed_covs = kalman_smoother(
       y_obs, data_map, ssm, params, filtered_states, filtered_covs, predicted_states, predicted_covs)
   
   # Forecast business conditions
   print(f"Forecasting {forecast_horizon} days ahead...")
   forecasts, forecast_covs = forecast_business_conditions(
       ssm, params, filtered_states, forecast_horizon)
   
   # Generate forecast dates
   last_date = data_map['dates'][-1]
   forecast_dates = pd.date_range(
       start=last_date + pd.Timedelta(days=1), 
       periods=forecast_horizon, 
       freq='D')
   
   # Plot results
   print("Plotting results...")
   plot_business_conditions(filtered_states, smoothed_states, data_map['dates'], 
                         true_factor, forecasts, forecast_dates)
   
   plot_indicators_with_factor(y_obs, data_map, smoothed_states)
   
   # Package results
   results = {
       'params': params,
       'ssm': ssm,
       'filtered_states': filtered_states,
       'filtered_covs': filtered_covs,
       'predicted_states': predicted_states,
       'predicted_covs': predicted_covs,
       'smoothed_states': smoothed_states,
       'smoothed_covs': smoothed_covs,
       'forecasts': forecasts,
       'forecast_covs': forecast_covs,
       'forecast_dates': forecast_dates,
       'data_map': data_map,
       'loglik': loglik
   }
   
   return results

# Main execution
# Set start and end dates
start_date = "2010-01-01"
end_date = "2019-12-31"

print("Generating simulated data...")
# Generate simulated data
data_dict, true_factor, factor_df = generate_simulated_data(start_date, end_date)

# Run the model
print("Running real-time business conditions measurement model...")
results = run_real_time_measurement(
    data_dict, 
    start_date=start_date, 
    end_date=end_date, 
    p=2,  # AR(2) untuk faktor laten
    trend_order=1,  # Trend linier
    true_factor=factor_df['true_factor'].values,
    forecast_horizon=90  # Prakiraan 90 hari
)

print("Analysis complete.")