Electricity Load Forecasting — Jersey Central Power & Light (JCPL)
A comparative deep learning study for short-term hourly electricity demand forecasting in the JCPL service area, using real operational data provided by Constellation Energy.
Overview
Accurate short-term electricity load forecasting is critical for real-time grid balancing, renewable energy integration, and optimized energy procurement. This project benchmarks classical statistical models against deep learning architectures on post-2022 JCPL hourly load data, with a focus on two rate classes: CIEP (commercial and industrial) and RSCP (residential and small commercial).
The core contribution is a region-specific comparative analysis trained and validated exclusively on JCPL's operational data — addressing the gap left by models built on aggregated national datasets that miss localized consumption patterns.
Course: DS 340W — Responsible Data Science, Penn State University (Fall 2025)
Authors: Ananya Drishti, Sree Penumuchu, Corrina Sigmund
Data Partner: Constellation Energy

Key Findings
ModelMAE (CIEP)RMSE (CIEP)MAE (RSCP)RMSE (RSCP)Linear Regression36.8 MW49.2 MW85.9 MW114.0 MWSARIMAX229.4 MW264.3 MW1310.0 MW1553.5 MWLSTM25.4 MW33.7 MW61.1 MW87.7 MWRNN24.2 MW32.8 MW58.7 MW84.7 MWCRNN (hybrid)19.5 MW25.2 MW59.1 MW80.4 MW

The hybrid CRNN reduced MAE ~23% over SARIMAX and ~12% over LSTM on the CIEP rate class
The RNN achieved R²=0.98 on CIEP across 5-fold cross-validation (mean MAE ~24 MW)
Seasonality is critical — without it, models flag predictable demand cycles as anomalies
SARIMAX degrades severely under high volatility; linear models systematically underpredict at temperature extremes


EDA Highlights

Temperature-load nonlinearity: U-shaped relationship with minimum load at 50–65°F; both heating and cooling drive demand at extremes
Weekday/weekend differential: CIEP weekend demand is 25–30% lower than weekday peaks, reflecting commercial/industrial usage patterns
Autocorrelation: Strong persistence at 1-hour (r=0.93), 24-hour (r=0.82), and 168-hour (r=0.76) lags — justifying lagged features in all models
Weather correlations: Pressure (0.22) and temperature (0.17) are the strongest weather predictors; wind speed and precipitation show minimal direct linear correlation


Models
Baseline Models

Multiple Linear Regression — interpretable benchmark; fails to capture nonlinear temperature effects
SARIMAX — explicit seasonality and autocorrelation modeling; collapses under volatile conditions

Deep Learning Models

LSTM — 64 hidden units, sequence length 24h, stacked architecture
RNN — 2-layer SimpleRNN (64→32 units) with lag features, rolling mean/std, dropout regularization
CRNN (hybrid) — Conv1D layers for short-term pattern extraction → GRU layers for long-range dependencies → rate class embedding → linear output

All deep learning models use:

5-fold walk-forward cross-validation
Adam optimizer, MSE loss, early stopping (patience=10)
Z-score normalization fitted on training set only


Data

Source: Constellation Energy (operational JCPL grid data) + NOAA weather stations
Coverage: January 2022 – October 2025 (~30,000 hourly records)
Rate classes: CIEP, RSCP, GPC, GPI, GSC, GSI, GSTC, GSTI, GTC, GTI, Residential
Features: Hourly load (MW), temperature, humidity, wind speed, precipitation, hour of day, day of week, weekend/holiday flags, lag features (t−1, t−24, t−168)


Note: The dataset was provided under a data-sharing agreement with Constellation Energy and is not included in this repository.


Repository Structure
├── JCPLdataFinalProjectCode.ipynb   # Full pipeline: EDA, feature engineering, all models
├── DS_340w_Term_Project.pdf         # Final written report
└── DS340W_Presentation.pdf          # Project presentation slides

Setup
bashpip install numpy pandas matplotlib scikit-learn tensorflow statsmodels

Individual Contributions

Sree Penumuchu — Weather data cleaning, full EDA (heatmaps, correlation matrices, autocorrelation analysis), RNN model development and cross-validation
Ananya Drishti — Linear Regression, SARIMAX, and CRNN model development; results analysis
Corrina Sigmund — Literature review, LSTM implementation, data inspection and validation
