SSO-GRNN Heat Pump Heat Supply Prediction

A complete Python implementation based on the paper *"Heat Supply Prediction for Heat Pump Systems Based on SSO-Optimized GRNN and Time Series Analysis"*, including data preprocessing, SSA denoising, dual correlation feature screening, SSO-optimized GRNN, and comprehensive experimental validation with multiple comparative models.



Complete Data Preprocessing Pipeline**: Linear interpolation for missing values, 3σ principle outlier detection and handling, Min-Max normalization
SSA Time Series Denoising and Reconstruction**: Singular Spectrum Analysis decomposes the original series, retaining effective components with cumulative contribution ≥ 95%
Dual Correlation Feature Screening**: Combines Pearson (linear) and Spearman (monotonic nonlinear) correlation analysis to select key features with |r|>0.3 and |ρ|>0.3
SSO-Optimized GRNN Model**: Sparrow Search Optimization optimizes the smoothing factor σ of Generalized Regression Neural Network
Multiple Comparative Model Implementations:
Traditional GRNN (fixed σ=0.1)
 Seasonal ARIMA
 LSTM
 PSO-optimized SVM
 SSO-optimized LSTM
Comprehensive Performance Evaluation System:
Core metrics: MAE, MSE, RMSE, MAPE, R²
Error distribution characteristic analysis
 Performance comparison under different volatility levels
  Seasonal adaptability analysis
  Noise interference robustness testing
  Morris screening method for feature sensitivity analysis
COP Prediction and Defrosting Period Analysis**: Additional implementation of heat pump system COP prediction with separate evaluation for defrosting/non-defrosting periods

Installation

Ensure you have Python 3.7+ installed, then install the required dependencies:

bash
pip install numpy pandas matplotlib scikit-learn scipy statsmodels tensorflow


Or use requirements.txt:
txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
scipy>=1.7.0
statsmodels>=0.13.0
tensorflow>=2.8.0


Usage

1. Prepare Data: Save your data in Excel format, ensuring it includes the following key columns (column names should match the paper):
Target column: `Hourly Heat Supply (kW)`
Feature columns: Ambient temperature, supply water temperature, return water temperature, system power, etc. (specific column names can be adjusted according to actual data, the code will automatically identify)

2. Configure File Path: Open the code and modify the filename variable on line 17:
python
   excel_file_name = 'your_data.xlsx'  # Replace with your Excel filename


3. Run the Code* Directly execute the Python script:
bash
   python sso_grnn_heat_pump_prediction.py


Code Structure

| Module | Description |
|--------|-------------|
| Data Loading & Preprocessing | Excel reading, missing value interpolation, outlier handling, normalization, defrosting period identification, COP calculation |
| `class SSA` | Singular Spectrum Analysis implementation, including trajectory matrix construction, SVD decomposition, diagonal averaging reconstruction |
| `class GRNN` | Generalized Regression Neural Network implementation, prediction based on Euclidean distance and Gaussian kernel function |
| `class SSO` | Sparrow Search Optimization implementation for optimizing GRNN smoothing factor and LSTM hyperparameters |
| `class PSO` | Particle Swarm Optimization implementation for optimizing SVM hyperparameters |
| `build_lstm_model` | Constructs a two-layer LSTM neural network model |
| Model Training & Prediction | Trains all comparative models and generates prediction results |
| Performance Evaluation Module | Comprehensive metric calculation, visualization data generation, and result output |


## License

MIT License
