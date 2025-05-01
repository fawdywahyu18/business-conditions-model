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

def prepare_data_for_model(data_dict, start_date, end_date, detrend=True):
    """
    Prepare the data for the prototype model implementation with detrending.
    
    Parameters:
    data_dict: Dictionary with keys as variable names and values as tuples of 
               (pandas Series with data, frequency, variable_type)
    start_date: datetime.date object for the start of the analysis
    end_date: datetime.date object for the end of the analysis
    detrend: Whether to detrend the series before estimation
    
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
        'lag_indices': {},  # Store indices of lagged observed values for AR(1)
        'dates': date_range,
        'max_Di': 0,  # Track maximum Di for state vector size
        'detrended': {}  # Store original and detrended series
    }
    
    y_obs = np.full((n_days, n_vars), np.nan)
    
    for i, (var_name, (series, freq, var_type)) in enumerate(data_dict.items()):
        data_map['frequencies'][var_name] = freq
        data_map['types'][var_name] = var_type
        
        # Detrend if requested
        if detrend:
            # Choose appropriate detrending method based on frequency
            if freq == 'Q':
                detrend_method = 'hp'  # Hodrick-Prescott for quarterly data
            else:
                detrend_method = 'linear'  # Linear for higher frequency data
            
            detrended_series = detrend_series(series, method=detrend_method)
            data_map['detrended'][var_name] = {
                'original': series,
                'detrended': detrended_series
            }
            # Use detrended series for model
            working_series = detrended_series
        else:
            working_series = series
        
        # Align data with our daily dates
        obs_indices = []
        for j, date in enumerate(tqdm(date_range, desc=f"Processing {var_name} data")):
            if date in working_series.index:
                y_obs[j, i] = working_series.loc[date]
                obs_indices.append(j)
        
        data_map['obs_indices'][var_name] = obs_indices
        
        # Find lag indices for AR(1) model
        lag_indices = {}
        if freq == 'W':  # Weekly
            # For each observation, find the previous week's observation
            for j in obs_indices:
                date = date_range[j]
                prev_week = date - pd.DateOffset(weeks=1)
                if prev_week in working_series.index:
                    lag_indices[j] = np.where(date_range == prev_week)[0][0]
        elif freq == 'M':  # Monthly
            # For each observation, find the previous month's observation
            for j in obs_indices:
                date = date_range[j]
                prev_month = date - pd.DateOffset(months=1)
                # Find the closest end-of-month date
                prev_month_end = pd.date_range(
                    start=prev_month.replace(day=1), 
                    end=prev_month + pd.DateOffset(day=31), 
                    freq='D')[-1]
                if prev_month_end in working_series.index:
                    lag_indices[j] = np.where(date_range == prev_month_end)[0][0]
        elif freq == 'Q':  # Quarterly
            # For each observation, find the previous quarter's observation
            for j in obs_indices:
                date = date_range[j]
                prev_quarter = date - pd.DateOffset(months=3)
                # Find the correct end-of-quarter date
                quarter_end_month = ((prev_quarter.month - 1) // 3) * 3 + 3
                
                # Calculate the last day of the month correctly
                if quarter_end_month in [4, 6, 9, 11]:  # April, June, September, November
                    last_day = 30
                elif quarter_end_month == 2:  # February
                    # Check for leap year
                    year = prev_quarter.year + (prev_quarter.month > 12)
                    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                        last_day = 29  # Leap year
                    else:
                        last_day = 28  # Non-leap year
                else:
                    last_day = 31
                
                # Create the end-of-quarter date
                year = prev_quarter.year
                if quarter_end_month > 12:
                    quarter_end_month = quarter_end_month - 12
                    year += 1
                
                prev_quarter_end = pd.Timestamp(
                    year=year,
                    month=quarter_end_month,
                    day=last_day
                )
                
                if prev_quarter_end in working_series.index:
                    lag_indices[j] = np.where(date_range == prev_quarter_end)[0][0]
        
        data_map['lag_indices'][var_name] = lag_indices
        
        # Calculate Di (days per period) for each observation
        if freq == 'D':
            data_map['Di'][var_name] = 1
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
def init_prototype_state_space(n_states, max_q):
    """Initialize state space matrices for the prototype model with numba for speed."""
    # Transition matrix T (equation 23)
    T = np.zeros((n_states, n_states))
    
    # AR(1) dynamics for x_t (first row)
    # Zero initially, will be filled with rho parameter later
    
    # Shifting for lags of x_t
    for i in range(1, max_q):
        T[i, i-1] = 1.0
    
    # AR(1) dynamics for u_t^1 (last row)
    # Zero initially, will be filled with gamma1 parameter later
    
    # Selection matrix R (equation 23)
    R = np.zeros((n_states, 2))
    R[0, 0] = 1.0  # Innovation to x_t
    R[-1, 1] = 1.0  # Innovation to u_t^1
    
    # Innovation covariance matrix Q (identity initially)
    Q = np.eye(2)
    
    # Initial state and covariance
    initial_state = np.zeros(n_states)
    initial_state_cov = np.eye(n_states) * 10.0
    
    return T, R, Q, initial_state, initial_state_cov

def construct_prototype_state_space(y_obs, data_map):
    """
    Construct the state-space model matrices according to equation (23).
    
    Parameters:
    y_obs: Observation array with NAs for missing values
    data_map: Data mapping information
    
    Returns:
    ssm: Dictionary containing state-space matrices
    """
    n_days, n_vars = y_obs.shape
    
    # Determine maximum number of days in a quarter (for state dimension)
    max_q = max(92, data_map['max_Di'])  # Maximum 92 days in a quarter + 1 for current day
    
    # State dimension: x_t and lags, plus u_t^1 for term premium
    n_states = max_q + 1  # current x_t, max_q-1 lags, and u_t^1
    
    # Initialize matrices with numba acceleration
    T, R, Q, initial_state, initial_state_cov = init_prototype_state_space(n_states, max_q)
    
    # Store dimensions and other information
    ssm = {
        'T': T,  # State transition matrix
        'R': R,  # Selection matrix
        'Q': Q,  # State innovation covariance
        'initial_state': initial_state,
        'initial_state_cov': initial_state_cov,
        'n_states': n_states,
        'n_obs': n_vars,
        'max_q': max_q
    }
    
    return ssm

@njit
def create_prototype_system_matrices(n_vars, n_states, max_q, t, variable_types, frequencies,
                                   obs_indices_array, lag_indices_values, lag_indices_keys,
                                   Di_values_array, Di_values_keys, beta, gamma, sigma):
    """
    Create the time-varying system matrices Z_t, Gamma_t, and H_t for the prototype model at time t.
    Menggunakan array 1D sebagai pengganti dictionary untuk kompatibilitas dengan Numba.
    
    Parameters:
    n_vars: Number of variables
    n_states: Number of state variables
    max_q: Maximum number of days in a quarter
    t: Time index
    variable_types: Array indicating variable types (0=stock, 1=flow)
    frequencies: Array indicating frequencies (0=D, 1=W, 2=M, 3=Q)
    obs_indices_array: Array of observed variables at time t
    lag_indices_values: Values of lag indices
    lag_indices_keys: Keys of lag indices (corresponding to observed variable indices)
    Di_values_array: Array of days-in-period values
    Di_values_keys: Keys of Di_values (corresponding to observed variable indices)
    beta: Array of beta coefficients
    gamma: Array of gamma coefficients
    sigma: Array of sigma coefficients
    
    Returns:
    Z_t: Observation matrix
    Gamma_t: Coefficient matrix for predetermined variables
    H_t: Observation innovation covariance
    w_t: Vector of predetermined variables
    """
    # Initialize matrices
    Z_t = np.zeros((n_vars, n_states))
    Gamma_t = np.zeros((n_vars, n_vars - 1))  # No lag for term premium
    H_t = np.zeros((n_vars, n_vars))
    w_t = np.zeros(n_vars - 1)
    
    # Term premium (daily, stock) - uses x_t and u_t^1
    Z_t[0, 0] = beta[0]  # Effect of x_t
    Z_t[0, -1] = 1.0     # Effect of u_t^1
    
    # Process each observed variable
    for i in range(len(obs_indices_array)):
        var_idx = obs_indices_array[i]
        
        # Skip term premium (already handled)
        if var_idx == 0:
            continue
            
        # Initial claims (weekly, flow)
        if var_idx == 1:
            # Find corresponding Di value
            days_in_week = 7  # Default
            for j in range(len(Di_values_keys)):
                if Di_values_keys[j] == var_idx:
                    days_in_week = Di_values_array[j]
                    break
                    
            # Sum of x_t over the week
            for j in range(min(days_in_week, max_q)):
                Z_t[var_idx, j] = beta[var_idx]
                
            # Lagged value in w_t (AR(1) component)
            for j in range(len(lag_indices_keys)):
                if lag_indices_keys[j] == var_idx:
                    Gamma_t[var_idx, 0] = gamma[var_idx]
                    w_t[0] = lag_indices_values[j]
                    break
                    
            # Measurement error variance scaled by days in week
            H_t[var_idx, var_idx] = sigma[var_idx]**2 * days_in_week
            
        # Employment (monthly, stock)
        elif var_idx == 2:
            Z_t[var_idx, 0] = beta[var_idx]  # Current value of x_t
            
            # Lagged value in w_t (AR(1) component)
            for j in range(len(lag_indices_keys)):
                if lag_indices_keys[j] == var_idx:
                    Gamma_t[var_idx, 1] = gamma[var_idx]
                    w_t[1] = lag_indices_values[j]
                    break
                    
            # Measurement error variance
            H_t[var_idx, var_idx] = sigma[var_idx]**2
            
        # GDP (quarterly, flow)
        elif var_idx == 3:
            # Find corresponding Di value
            days_in_quarter = 90  # Default
            for j in range(len(Di_values_keys)):
                if Di_values_keys[j] == var_idx:
                    days_in_quarter = Di_values_array[j]
                    break
                    
            # Sum of x_t over the quarter
            for j in range(min(days_in_quarter, max_q)):
                Z_t[var_idx, j] = beta[var_idx]
                
            # Lagged value in w_t (AR(1) component)
            for j in range(len(lag_indices_keys)):
                if lag_indices_keys[j] == var_idx:
                    Gamma_t[var_idx, 2] = gamma[var_idx]
                    w_t[2] = lag_indices_values[j]
                    break
                    
            # Measurement error variance scaled by days in quarter
            H_t[var_idx, var_idx] = sigma[var_idx]**2 * days_in_quarter
    
    return Z_t, Gamma_t, H_t, w_t

@njit
def kalman_filter_step(a_t, P_t, y_t, Z_t, Gamma_t, H_t, T, R, Q, w_t, obs_vector):
   """Perform one step of the Kalman filter with numba acceleration."""
   # Store the predicted state and covariance
   predicted_state = a_t.copy()
   predicted_cov = P_t.copy()
   
   # Check if any variables are observed
   if np.any(obs_vector):
       # Transform system to handle missing data
       y_t_star, Z_t_star, Gamma_t_star, H_t_star, * = transform_system_numba(
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
   Run the Kalman filter for the prototype model with progress bar.
   
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
   
   # Set up transition matrix with AR(1) parameters
   T = ssm['T'].copy()
   T[0, 0] = params['rho']      # AR(1) for business conditions
   T[-1, -1] = params['gamma'][0]  # AR(1) for term premium error
   
   # Set up covariance matrix for innovations
   Q = ssm['Q'].copy()
   Q[0, 0] = 1.0                 # Normalized to 1
   Q[1, 1] = params['sigma'][0]**2  # Variance of term premium error
   
   # Kalman filter recursion with progress bar
   for t in tqdm(range(n_days), desc="Running Kalman filter"):
       # Create time-varying system matrices
       Z_t, Gamma_t, H_t, w_t, obs_vector = create_system_matrices_for_prototype(
           ssm, data_map, t, params, y_obs)  # Passing y_obs as an additional parameter
       
       # Extract observation vector for time t
       y_t = y_obs[t]
       
       # Perform one step of the Kalman filter
       a_t_t, P_t_t, a_t_next, P_t_next, loglik_t, valid_update = kalman_filter_step(
           a_t, P_t, y_t, Z_t, Gamma_t, H_t, T, ssm['R'], Q, w_t, obs_vector)
       
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
