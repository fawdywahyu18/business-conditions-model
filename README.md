# business-conditions-model

A state-space framework for real-time measurement of economic conditions using mixed-frequency data.

## Description

This repository implements the approach proposed by Aruoba, Diebold, and Scotti in their paper "Real-Time Measurement of Business Conditions." The model treats business conditions as an unobserved dynamic factor that can be extracted from indicators measured at different frequencies.

## Key Features

- **Dynamic Factor Framework**: Treats business conditions as a latent variable
- **Mixed-Frequency Support**: Properly handles indicators at daily, weekly, monthly, and quarterly frequencies
- **Stock and Flow Variables**: Correctly distinguishes between stock and flow variables
- **Exact Filtering**: Uses statistically optimal Kalman filter/smoother without approximations
- **Simulation Capabilities**: Generate realistic economic data with known parameters
- **Validation Tools**: Verify theoretical consistency of estimated models
- **Forecasting**: Project business conditions forward using estimated dynamics

## Implementation Details

The code includes:
- Optimized Kalman filtering with numba acceleration
- Parameter estimation via maximum likelihood
- Handling of missing observations and mixed frequencies
- Detrending options for various data types
- Real-time measurement and forecasting capabilities

## References

Aruoba, S.B., Diebold, F.X., and Scotti, C. (2009). "Real-Time Measurement of Business Conditions." *Journal of Business & Economic Statistics*, 27(4), 417-427.
