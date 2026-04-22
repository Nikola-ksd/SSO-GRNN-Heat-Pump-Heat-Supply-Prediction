import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from scipy.linalg import svd
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

excel_file_name = ''
df = pd.read_excel(excel_file_name)

print('=' * 80)
print('Data Loading and Initial Inspection')
print('=' * 80)
print(f'Dataset shape: {df.shape}')
print(f'Dataset columns: {list(df.columns)}')
print(f'First 10 rows:')
print(df.head(10))
print(f'Dataset info:')
print(df.info())
print(f'Dataset description:')
print(df.describe())
print()

target_col = 'Hourly Heat Supply (kW)'
potential_feature_cols = [col for col in df.columns if col != target_col]

print('=' * 80)
print('Defrosting Operation Identification')
print('=' * 80)


def identify_defrosting(df):
    defrost_mask = np.zeros(len(df), dtype=bool)
    required_cols = ['Supply Water Temperature', 'Return Water Temperature', 'Ambient Temperature',
                     'Unit Operating Voltage', 'Unit Current', 'Heat Pump Power']
    available_cols = [col for col in required_cols if col in df.columns]
    if len(available_cols) >= 3:
        supply_temp = df[available_cols[0]].values
        return_temp = df[available_cols[1]].values
        power = df[available_cols[-1]].values
        temp_drop = (supply_temp < np.roll(supply_temp, 1) - 2) | (return_temp < np.roll(return_temp, 1) - 2)
        power_fluct = np.abs(power - np.roll(power, 1)) > 0.3 * np.roll(power, 1)
        defrost_mask = temp_drop & power_fluct
        defrost_mask[0] = False
    df['Defrosting'] = defrost_mask
    print(f'Defrosting periods identified: {np.sum(defrost_mask)} hours')
    return df


df = identify_defrosting(df)
print()

print('=' * 80)
print('Target Variable and COP Calculation')
print('=' * 80)

print(f'Target column: {target_col}')
print(f'Target range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]')
print(f'Target mean: {df[target_col].mean():.2f}')
print(f'Target median: {df[target_col].median():.2f}')
print(f'Target std: {df[target_col].std():.2f}')

power_col = None
for col in df.columns:
    if 'power' in col.lower() and 'heat' in col.lower():
        power_col = col
        break
if power_col is None:
    for col in df.columns:
        if 'power' in col.lower():
            power_col = col
            break

if power_col is not None:
    df['COP'] = df[target_col] / df[power_col]
    df['COP'] = df['COP'].clip(2.2, 4.8)
    print(f'COP calculated using {power_col}')
    print(f'COP range: [{df["COP"].min():.2f}, {df["COP"].max():.2f}]')
    print(f'COP mean: {df["COP"].mean():.2f}')
    print(f'COP std: {df["COP"].std():.2f}')
else:
    df['COP'] = np.nan
    print('No suitable power column found for COP calculation')
print()

feature_cols = [col for col in potential_feature_cols if col not in ['Defrosting']]
X = df[feature_cols].values
y = df[target_col].values
cop = df['COP'].values

print('=' * 80)
print('Missing Value Imputation')
print('=' * 80)
missing_X = np.isnan(X).sum()
missing_y = np.isnan(y).sum()
missing_cop = np.isnan(cop).sum()
print(f'Original missing values in features: {missing_X}')
print(f'Original missing values in target: {missing_y}')
print(f'Original missing values in COP: {missing_cop}')

X = np.apply_along_axis(
    lambda col: np.interp(np.arange(len(col)), np.arange(len(col))[~np.isnan(col)], col[~np.isnan(col)]), axis=0, arr=X)
y = np.interp(np.arange(len(y)), np.arange(len(y))[~np.isnan(y)], y[~np.isnan(y)])
cop = np.interp(np.arange(len(cop)), np.arange(len(cop))[~np.isnan(cop)], cop[~np.isnan(cop)])

missing_X_after = np.isnan(X).sum()
missing_y_after = np.isnan(y).sum()
missing_cop_after = np.isnan(cop).sum()
print(f'Missing values in features after imputation: {missing_X_after}')
print(f'Missing values in target after imputation: {missing_y_after}')
print(f'Missing values in COP after imputation: {missing_cop_after}')
print()

print('=' * 80)
print('Outlier Detection and Removal')
print('=' * 80)


def detect_outliers(data):
    mean = np.mean(data)
    std = np.std(data)
    lower = mean - 3 * std
    upper = mean + 3 * std
    outliers = (data < lower) | (data > upper)
    return outliers, lower, upper


total_outliers_X = 0
for i in range(X.shape[1]):
    outliers, lower, upper = detect_outliers(X[:, i])
    num_outliers = np.sum(outliers)
    total_outliers_X += num_outliers
    if num_outliers > 0:
        print(f'Feature {feature_cols[i]}: {num_outliers} outliers detected, range [{lower:.2f}, {upper:.2f}]')
        X[outliers, i] = np.mean(X[:, i])

outliers_y, lower_y, upper_y = detect_outliers(y)
num_outliers_y = np.sum(outliers_y)
print(f'Target variable: {num_outliers_y} outliers detected, range [{lower_y:.2f}, {upper_y:.2f}]')
y[outliers_y] = np.mean(y)

outliers_cop, lower_cop, upper_cop = detect_outliers(cop)
num_outliers_cop = np.sum(outliers_cop)
print(f'COP: {num_outliers_cop} outliers detected, range [{lower_cop:.2f}, {upper_cop:.2f}]')
cop[outliers_cop] = np.mean(cop)

print(f'Total outliers in features: {total_outliers_X}')
print(f'Total outliers in target: {num_outliers_y}')
print(f'Total outliers in COP: {num_outliers_cop}')
print()

print('=' * 80)
print('Data Normalization')
print('=' * 80)

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_cop = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
cop_scaled = scaler_cop.fit_transform(cop.reshape(-1, 1)).flatten()

print(f'Features normalized to [0, 1]')
print(f'Target normalized to [0, 1]')
print(f'COP normalized to [0, 1]')
print(f'Normalized features range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]')
print(f'Normalized target range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]')
print(f'Normalized COP range: [{cop_scaled.min():.4f}, {cop_scaled.max():.4f}]')
print()


class SSA:
    def __init__(self, L=None):
        self.L = L

    def fit(self, X):
        self.X = X
        N = len(X)
        if self.L is None:
            self.L = N // 3
        self.K = N - self.L + 1
        self.Y = self._create_trajectory_matrix()
        self.U, self.Sigma, self.VT = svd(self.Y, full_matrices=False)
        self.lambdas = self.Sigma ** 2
        self.contributions = self.lambdas / self.lambdas.sum()
        self.cumulative_contributions = np.cumsum(self.contributions)
        return self

    def _create_trajectory_matrix(self):
        X = self.X
        L = self.L
        K = self.K
        Y = np.zeros((L, K))
        for i in range(K):
            Y[:, i] = X[i:i + L]
        return Y

    def reconstruct(self, m=None):
        if m is None:
            m = np.where(self.cumulative_contributions >= 0.95)[0][0] + 1
        U_m = self.U[:, :m]
        Sigma_m = np.diag(self.Sigma[:m])
        VT_m = self.VT[:m, :]
        Y_m = U_m @ Sigma_m @ VT_m
        X_rec = self._diagonal_averaging(Y_m)
        return X_rec, m

    def _diagonal_averaging(self, Y):
        L, K = Y.shape
        N = L + K - 1
        X_rec = np.zeros(N)
        for i in range(N):
            if i < L:
                start_j = 0
                end_j = i + 1
                count = i + 1
            elif i < K:
                start_j = i - L + 1
                end_j = i + 1
                count = L
            else:
                start_j = i - L + 1
                end_j = K
                count = N - i
            for j in range(start_j, end_j):
                X_rec[i] += Y[i - j, j]
            X_rec[i] /= count
        return X_rec


print('=' * 80)
print('SSA-Based Denoising Reconstruction')
print('=' * 80)

ssa = SSA()
ssa.fit(y_scaled)
y_rec, m = ssa.reconstruct()

print(f'Embedding dimension L: {ssa.L}')
print(f'Trajectory matrix shape: {ssa.Y.shape}')
print(f'Number of singular values: {len(ssa.Sigma)}')
print(f'Singular values (top 10): {ssa.Sigma[:10]}')
print(f'Contribution rates (top 10): {ssa.contributions[:10]}')
print(f'Cumulative contribution rates (top 10): {ssa.cumulative_contributions[:10]}')
print(f'Selected components for reconstruction: {m}')
print(f'Cumulative contribution rate: {ssa.cumulative_contributions[m - 1]:.4f}')
print()

print('=' * 80)
print('Dual Correlation Analysis for Feature Screening')
print('=' * 80)

pearson_corr = []
spearman_corr = []
p_values_pearson = []
p_values_spearman = []

for i in range(X_scaled.shape[1]):
    r, p_r = pearsonr(X_scaled[:, i], y_rec)
    rho, p_rho = spearmanr(X_scaled[:, i], y_rec)
    pearson_corr.append(abs(r))
    spearman_corr.append(abs(rho))
    p_values_pearson.append(p_r)
    p_values_spearman.append(p_rho)
    print(
        f'Feature {feature_cols[i]}: Pearson |r| = {abs(r):.4f} (p={p_r:.4f}), Spearman |rho| = {abs(rho):.4f} (p={p_rho:.4f})')

pearson_corr = np.array(pearson_corr)
spearman_corr = np.array(spearman_corr)
selected_features = np.where((pearson_corr > 0.3) & (spearman_corr > 0.3))[0]
X_selected = X_scaled[:, selected_features]
selected_feature_names = [feature_cols[k] for k in selected_features]

print()
print(f'Selected features (|r|>0.3 and |rho|>0.3): {selected_feature_names}')
print(f'Number of selected features: {len(selected_features)}')
print(f'Original feature dimension: {X_scaled.shape[1]}')
print(f'Reduced feature dimension: {X_selected.shape[1]}')
print()

X_final = np.hstack((X_selected, y_rec.reshape(-1, 1)))
final_feature_names = selected_feature_names + ['SSA_Reconstructed_Component']

print(f'Final input feature set shape: {X_final.shape}')
print(f'Final features: {final_feature_names}')
print()

print('=' * 80)
print('Dataset Splitting')
print('=' * 80)

N = len(y_scaled)
train_size = int(0.7 * N)
X_train, X_test = X_final[:train_size], X_final[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
cop_train, cop_test = cop_scaled[:train_size], cop_scaled[train_size:]
defrost_test = df['Defrosting'].values[train_size:]

val_size = int(0.2 * len(X_train))
X_train_sub, X_val = X_train[:-val_size], X_train[-val_size:]
y_train_sub, y_val = y_train[:-val_size], y_train[-val_size:]

print(f'Total samples: {N}')
print(f'Training samples: {len(X_train)} ({len(X_train) / N * 100:.1f}%)')
print(f'Validation samples: {len(X_val)} ({len(X_val) / N * 100:.1f}%)')
print(f'Test samples: {len(X_test)} ({len(X_test) / N * 100:.1f}%)')
print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print()


class GRNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        y_pred = np.zeros(len(X_test))
        for i, x in enumerate(X_test):
            distances = np.linalg.norm(self.X_train - x, axis=1)
            weights = np.exp(-distances ** 2 / (2 * self.sigma ** 2))
            if weights.sum() == 0:
                y_pred[i] = self.y_train.mean()
            else:
                y_pred[i] = (weights @ self.y_train) / weights.sum()
        return y_pred


class SSO:
    def __init__(self, pop_size=30, max_iter=100, producer_ratio=0.2, scrounger_ratio=0.1, lb=0.01, ub=1):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.producer_ratio = producer_ratio
        self.scrounger_ratio = scrounger_ratio
        self.lb = lb
        self.ub = ub
        self.dim = 1
        self.fitness_history = []

    def fit(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.array([self._compute_fitness(sigma) for sigma in self.pop])
        self.best_idx = np.argmin(self.fitness)
        self.best_sigma = self.pop[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        self.worst_idx = np.argmax(self.fitness)
        self.worst_sigma = self.pop[self.worst_idx].copy()
        self.worst_fitness = self.fitness[self.worst_idx]
        self.fitness_history.append(self.best_fitness)

        for t in range(self.max_iter):
            self._update_producers()
            self._update_followers()
            self._update_scroungers()
            self.fitness = np.array([self._compute_fitness(sigma) for sigma in self.pop])
            current_best_idx = np.argmin(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]
            if current_best_fitness < self.best_fitness:
                self.best_sigma = self.pop[current_best_idx].copy()
                self.best_fitness = current_best_fitness
            current_worst_idx = np.argmax(self.fitness)
            current_worst_fitness = self.fitness[current_worst_idx]
            if current_worst_fitness > self.worst_fitness:
                self.worst_sigma = self.pop[current_worst_idx].copy()
                self.worst_fitness = current_worst_fitness
            self.fitness_history.append(self.best_fitness)
            if (t + 1) % 10 == 0:
                print(f'Iteration {t + 1}/{self.max_iter}, Best Fitness (MAPE): {self.best_fitness:.4f}%')

        return self.best_sigma[0]

    def _compute_fitness(self, sigma):
        grnn = GRNN(sigma=sigma[0])
        grnn.fit(self.X_train, self.y_train)
        y_pred = grnn.predict(self.X_val)
        mape = np.mean(np.abs((self.y_val - y_pred) / (self.y_val + 1e-6))) * 100
        return mape

    def _update_producers(self):
        num_producers = int(self.pop_size * self.producer_ratio)
        producer_indices = np.argsort(self.fitness)[:num_producers]
        ST = 0.8
        for i in producer_indices:
            r1 = np.random.rand()
            if r1 < ST:
                self.pop[i] = self.pop[i] * np.exp(-i / (np.random.rand() * self.max_iter + 1e-6))
            else:
                self.pop[i] = self.pop[i] + np.random.randn() * np.ones(self.dim)
            self.pop[i] = np.clip(self.pop[i], self.lb, self.ub)

    def _update_followers(self):
        num_producers = int(self.pop_size * self.producer_ratio)
        follower_indices = np.argsort(self.fitness)[num_producers:]
        for i, idx in enumerate(follower_indices):
            if i >= len(follower_indices) // 2:
                self.pop[idx] = np.random.randn() * np.exp(-(self.worst_sigma - self.pop[idx]) ** 2 / 2)
            else:
                X_p = self.best_sigma
                A = np.random.choice([-1, 1], self.dim)
                A_plus = A.T / (A @ A.T + 1e-6)
                self.pop[idx] = X_p + np.abs(np.random.randn() * np.ones(self.dim)) * (A_plus * np.ones(self.dim))
            self.pop[idx] = np.clip(self.pop[idx], self.lb, self.ub)

    def _update_scroungers(self):
        num_scroungers = int(self.pop_size * self.scrounger_ratio)
        scrounger_indices = np.random.choice(self.pop_size, num_scroungers, replace=False)
        ST = 0.8
        for i in scrounger_indices:
            r2 = np.random.rand()
            if r2 < ST:
                self.pop[i] = self.best_sigma + np.random.randn() * np.abs(self.pop[i] - self.best_sigma)
            else:
                K = np.random.uniform(-1, 1, self.dim)
                self.pop[i] = self.pop[i] + K * (
                            np.abs(self.pop[i] - self.worst_sigma) / ((self.pop[i] - self.worst_sigma) ** 2 + 1e-6))
            self.pop[i] = np.clip(self.pop[i], self.lb, self.ub)


class PSO:
    def __init__(self, pop_size=30, max_iter=100, lb=1, ub=100, lb_gamma=0.01, ub_gamma=10):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.array([lb, lb_gamma])
        self.ub = np.array([ub, ub_gamma])
        self.dim = 2
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

    def fit(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.velocities = np.zeros_like(self.pop)
        self.pbest_pos = self.pop.copy()
        self.pbest_fitness = np.array([self._compute_fitness(params) for params in self.pop])
        self.gbest_idx = np.argmin(self.pbest_fitness)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_fitness = self.pbest_fitness[self.gbest_idx]

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_pos[i] - self.pop[i]) +
                                      self.c2 * r2 * (self.gbest_pos - self.pop[i]))
                self.pop[i] = self.pop[i] + self.velocities[i]
                self.pop[i] = np.clip(self.pop[i], self.lb, self.ub)
                fitness = self._compute_fitness(self.pop[i])
                if fitness < self.pbest_fitness[i]:
                    self.pbest_pos[i] = self.pop[i].copy()
                    self.pbest_fitness[i] = fitness
                    if fitness < self.gbest_fitness:
                        self.gbest_pos = self.pop[i].copy()
                        self.gbest_fitness = fitness
            if (t + 1) % 10 == 0:
                print(f'PSO Iteration {t + 1}/{self.max_iter}, Best Fitness (MAPE): {self.gbest_fitness:.4f}%')

        return self.gbest_pos

    def _compute_fitness(self, params):
        C, gamma = params
        svr = SVR(C=C, gamma=gamma, kernel='rbf')
        svr.fit(self.X_train, self.y_train)
        y_pred = svr.predict(self.X_val)
        mape = np.mean(np.abs((self.y_val - y_pred) / (self.y_val + 1e-6))) * 100
        return mape


def build_lstm_model(input_shape, units=64, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model


print('=' * 80)
print('SSO Optimization for SSO-GRNN and SSO-LSTM')
print('=' * 80)

sso = SSO(pop_size=30, max_iter=100, producer_ratio=0.2, scrounger_ratio=0.1, lb=0.01, ub=1)
best_sigma = sso.fit(X_train_sub, y_train_sub, X_val, y_val)

print()
print(f'SSO-GRNN Optimization complete!')
print(f'Optimal smoothing factor sigma: {best_sigma:.6f}')
print(f'Best fitness (MAPE on validation set): {sso.best_fitness:.4f}%')
print()


class SSO_LSTM_Optimizer(SSO):
    def __init__(self, pop_size=30, max_iter=50, producer_ratio=0.2, scrounger_ratio=0.1,
                 lb_units=32, ub_units=128, lb_lr=0.0001, ub_lr=0.01):
        super().__init__(pop_size, max_iter, producer_ratio, scrounger_ratio, lb=0, ub=1)
        self.lb_units = lb_units
        self.ub_units = ub_units
        self.lb_lr = lb_lr
        self.ub_lr = ub_lr
        self.dim = 2

    def _decode_params(self, params):
        units = int(self.lb_units + params[0] * (self.ub_units - self.lb_units))
        lr = self.lb_lr + params[1] * (self.ub_lr - self.lb_lr)
        return units, lr

    def _compute_fitness(self, params):
        units, lr = self._decode_params(params)
        X_train_reshaped = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        X_val_reshaped = self.X_val.reshape((self.X_val.shape[0], 1, self.X_val.shape[1]))
        model = build_lstm_model((1, self.X_train.shape[1]), units=units, learning_rate=lr)
        model.fit(X_train_reshaped, self.y_train, epochs=20, batch_size=32, verbose=0)
        y_pred = model.predict(X_val_reshaped, verbose=0).flatten()
        mape = np.mean(np.abs((self.y_val - y_pred) / (self.y_val + 1e-6))) * 100
        return mape


sso_lstm_opt = SSO_LSTM_Optimizer(pop_size=20, max_iter=30)
best_lstm_params = sso_lstm_opt.fit(X_train_sub, y_train_sub, X_val, y_val)
best_units, best_lr = sso_lstm_opt._decode_params(best_lstm_params)

print()
print(f'SSO-LSTM Optimization complete!')
print(f'Optimal units: {best_units}, Optimal learning rate: {best_lr:.6f}')
print(f'Best fitness (MAPE on validation set): {sso_lstm_opt.best_fitness:.4f}%')
print()

print('=' * 80)
print('Training All Models')
print('=' * 80)

models = {}
predictions = {}

print('Training Traditional GRNN...')
traditional_grnn = GRNN(sigma=0.1)
traditional_grnn.fit(X_train, y_train)
models['Traditional GRNN'] = traditional_grnn

print('Training Seasonal ARIMA...')
try:
    sarima = SARIMAX(y_train, order=(2, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima.fit(disp=False)
    models['Seasonal ARIMA'] = sarima_fit
except Exception as e:
    print(f'SARIMA training failed: {e}')
    models['Seasonal ARIMA'] = None

print('Training LSTM...')
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
lstm_model = build_lstm_model((1, X_train.shape[1]), units=64, learning_rate=0.001)
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=0)
models['LSTM'] = lstm_model

print('Training PSO-SVM...')
pso = PSO(pop_size=30, max_iter=100)
best_svr_params = pso.fit(X_train_sub, y_train_sub, X_val, y_val)
best_C, best_gamma = best_svr_params
pso_svr = SVR(C=best_C, gamma=best_gamma, kernel='rbf')
pso_svr.fit(X_train, y_train)
models['PSO-SVM'] = pso_svr

print('Training SSO-LSTM...')
sso_lstm = build_lstm_model((1, X_train.shape[1]), units=best_units, learning_rate=best_lr)
sso_lstm.fit(X_train_lstm, y_train, epochs=100, batch_size=32, verbose=0)
models['SSO-LSTM'] = sso_lstm

print('Training SSO-GRNN (Proposed)...')
sso_grnn = GRNN(sigma=best_sigma)
sso_grnn.fit(X_train, y_train)
models['SSO-GRNN (Proposed)'] = sso_grnn

print()
print('=' * 80)
print('Generating Predictions for All Models')
print('=' * 80)

for name, model in models.items():
    if model is None:
        predictions[name] = np.full(len(y_test), np.nan)
        continue
    print(f'Predicting with {name}...')
    if name == 'Seasonal ARIMA':
        pred_scaled = model.get_forecast(steps=len(y_test)).predicted_mean
    elif name in ['LSTM', 'SSO-LSTM']:
        pred_scaled = model.predict(X_test_lstm, verbose=0).flatten()
    else:
        pred_scaled = model.predict(X_test)
    predictions[name] = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
print()

print('=' * 80)
print('Overall Prediction Performance Comparison')
print('=' * 80)
print(f'{"Model":<25} {"MAE (kW)":<12} {"MSE (kW²)":<12} {"RMSE (kW)":<12} {"MAPE (%)":<12} {"R²":<10}')
print('-' * 85)

results = []
for name in models.keys():
    y_pred = predictions[name]
    if np.isnan(y_pred).all():
        print(f'{name:<25} {"N/A":<12} {"N/A":<12} {"N/A":<12} {"N/A":<12} {"N/A":<10}')
        continue
    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_orig - y_pred) / (y_test_orig + 1e-6))) * 100
    r2 = r2_score(y_test_orig, y_pred)
    results.append((name, mae, mse, rmse, mape, r2))
    print(f'{name:<25} {mae:<12.2f} {mse:<12.2f} {rmse:<12.2f} {mape:<12.2f} {r2:<10.4f}')
print()

print('=' * 80)
print('Error Distribution Characteristic Analysis (SSO-GRNN)')
print('=' * 80)

y_pred_sso = predictions['SSO-GRNN (Proposed)']
errors = y_test_orig - y_pred_sso
error_mean = np.mean(errors)
error_std = np.std(errors)
error_min = np.min(errors)
error_max = np.max(errors)
error_median = np.median(errors)

within_05kw = np.sum(np.abs(errors) <= 0.5) / len(errors) * 100
within_10kw = np.sum(np.abs(errors) <= 1.0) / len(errors) * 100
within_15kw = np.sum(np.abs(errors) <= 1.5) / len(errors) * 100
within_20kw = np.sum(np.abs(errors) <= 2.0) / len(errors) * 100
extreme_errors = np.sum(np.abs(errors) > 3) / len(errors) * 100

print(f'Error statistics:')
print(f'  Mean: {error_mean:.4f} kW')
print(f'  Std: {error_std:.4f} kW')
print(f'  Min: {error_min:.4f} kW')
print(f'  Max: {error_max:.4f} kW')
print(f'  Median: {error_median:.4f} kW')
print()
print(f'Error distribution:')
print(f'  Within [-0.5, 0.5] kW: {within_05kw:.1f}%')
print(f'  Within [-1.0, 1.0] kW: {within_10kw:.1f}%')
print(f'  Within [-1.5, 1.5] kW: {within_15kw:.1f}%')
print(f'  Within [-2.0, 2.0] kW: {within_20kw:.1f}%')
print(f'  Extreme errors (>3 kW or <-3 kW): {extreme_errors:.1f}%')
print()

print('=' * 80)
print('Performance Under Different Volatility Levels (All Models)')
print('=' * 80)

volatility = np.abs(np.diff(y_test_orig, prepend=y_test_orig[0])) / (y_test_orig + 1e-6) * 100
low_vol_idx = volatility < 2
med_vol_idx = (volatility >= 2) & (volatility < 5)
high_vol_idx = volatility >= 5


def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100


print(f'{"Model":<25} {"Low Vol (<2%)":<15} {"Med Vol (2-5%)":<15} {"High Vol (≥5%)":<15} {"Average":<10}')
print('-' * 80)

for name in models.keys():
    y_pred = predictions[name]
    if np.isnan(y_pred).all():
        print(f'{name:<25} {"N/A":<15} {"N/A":<15} {"N/A":<15} {"N/A":<10}')
        continue
    mape_low = compute_mape(y_test_orig[low_vol_idx], y_pred[low_vol_idx])
    mape_med = compute_mape(y_test_orig[med_vol_idx], y_pred[med_vol_idx])
    mape_high = compute_mape(y_test_orig[high_vol_idx], y_pred[high_vol_idx])
    mape_avg = compute_mape(y_test_orig, y_pred)
    print(f'{name:<25} {mape_low:<15.2f} {mape_med:<15.2f} {mape_high:<15.2f} {mape_avg:<10.2f}')
print()

print('=' * 80)
print('Seasonal Operating Condition Adaptability (SSO-GRNN)')
print('=' * 80)

if 'Datetime' in df.columns:
    test_time = pd.to_datetime(df['Datetime'].values[train_size:])
else:
    test_time = pd.date_range(start='2022-01-01', periods=len(y_test_orig), freq='H')

winter_idx = test_time.month == 1
spring_idx = (test_time.month == 2) | (test_time.month == 3)


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = compute_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2


if np.sum(winter_idx) > 0:
    mae_winter, rmse_winter, mape_winter, r2_winter = compute_metrics(y_test_orig[winter_idx], y_pred_sso[winter_idx])
    print('Winter (January):')
    print(
        f'  Samples: {np.sum(winter_idx)}, MAE: {mae_winter:.2f} kW, RMSE: {rmse_winter:.2f} kW, MAPE: {mape_winter:.2f}%, R²: {r2_winter:.4f}')

if np.sum(spring_idx) > 0:
    mae_spring, rmse_spring, mape_spring, r2_spring = compute_metrics(y_test_orig[spring_idx], y_pred_sso[spring_idx])
    print('Spring (February-March):')
    print(
        f'  Samples: {np.sum(spring_idx)}, MAE: {mae_spring:.2f} kW, RMSE: {rmse_spring:.2f} kW, MAPE: {mape_spring:.2f}%, R²: {r2_spring:.4f}')
print()

print('=' * 80)
print('Robustness Analysis Under Noise Interference (All Models)')
print('=' * 80)

noise_levels = [0.05, 0.10, 0.15, 0.20]
noise_results = {name: [] for name in models.keys()}

for noise in noise_levels:
    print(f'Testing with {int(noise * 100)}% noise...')
    X_test_noisy = X_test + noise * np.random.randn(*X_test.shape)
    X_test_noisy_lstm = X_test_noisy.reshape((X_test_noisy.shape[0], 1, X_test_noisy.shape[1]))
    for name, model in models.items():
        if model is None:
            noise_results[name].append(np.nan)
            continue
        if name == 'Seasonal ARIMA':
            noise_results[name].append(np.nan)
        elif name in ['LSTM', 'SSO-LSTM']:
            pred_scaled = model.predict(X_test_noisy_lstm, verbose=0).flatten()
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            noise_results[name].append(compute_mape(y_test_orig, pred))
        else:
            pred_scaled = model.predict(X_test_noisy)
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            noise_results[name].append(compute_mape(y_test_orig, pred))

print()
print(f'{"Model":<25} {"5% Noise":<12} {"10% Noise":<12} {"15% Noise":<12} {"20% Noise":<12}')
print('-' * 75)

for name in models.keys():
    mape_values = noise_results[name]
    print(f'{name:<25}', end='')
    for mp in mape_values:
        if np.isnan(mp):
            print(f' {"N/A":<11}', end='')
        else:
            print(f' {mp:<11.2f}', end='')
    print()
print()

print('=' * 80)
print('Feature Sensitivity Analysis (Morris Screening Method - SSO-GRNN)')
print('=' * 80)

num_samples = 200
num_levels = 4
sensitivity_indices = np.zeros(X_final.shape[1])

for i in range(X_final.shape[1]):
    delta = np.zeros((num_samples, X_final.shape[1]))
    base_values = np.random.uniform(0, 1, (num_samples, X_final.shape[1]))
    for j in range(num_samples):
        base = base_values[j].copy()
        level = np.random.randint(num_levels)
        delta[j] = base
        delta[j, i] = (base[i] + (level + 1) / num_levels) % 1
    y_base = sso_grnn.predict(base_values)
    y_delta = sso_grnn.predict(delta)
    sensitivity_indices[i] = np.mean(np.abs(y_delta - y_base))

sensitivity_indices = sensitivity_indices / sensitivity_indices.sum()
sorted_idx = np.argsort(sensitivity_indices)[::-1]

print('Feature sensitivity indices (sorted by importance):')
for idx in sorted_idx:
    print(f'  {final_feature_names[idx]}: {sensitivity_indices[idx]:.4f}')
print()

print('=' * 80)
print('COP Prediction Performance Comparison')
print('=' * 80)

cop_models = {}
cop_predictions = {}

print('Training COP models...')
X_train_cop, X_test_cop = X_final[:train_size], X_final[train_size:]

print('  Traditional GRNN for COP...')
cop_traditional_grnn = GRNN(sigma=0.1)
cop_traditional_grnn.fit(X_train_cop, cop_train)
cop_models['Traditional GRNN'] = cop_traditional_grnn

print('  LSTM for COP...')
cop_lstm = build_lstm_model((1, X_train_cop.shape[1]), units=64, learning_rate=0.001)
cop_lstm.fit(X_train_cop.reshape((-1, 1, X_train_cop.shape[1])), cop_train, epochs=100, batch_size=32, verbose=0)
cop_models['LSTM'] = cop_lstm

print('  PSO-SVM for COP...')
cop_pso = PSO(pop_size=30, max_iter=50)
cop_best_svr_params = cop_pso.fit(X_train_sub, y_train_sub, X_val, y_val)
cop_best_C, cop_best_gamma = cop_best_svr_params
cop_pso_svr = SVR(C=cop_best_C, gamma=cop_best_gamma, kernel='rbf')
cop_pso_svr.fit(X_train_cop, cop_train)
cop_models['PSO-SVM'] = cop_pso_svr

print('  SSO-LSTM for COP...')
cop_sso_lstm = build_lstm_model((1, X_train_cop.shape[1]), units=best_units, learning_rate=best_lr)
cop_sso_lstm.fit(X_train_cop.reshape((-1, 1, X_train_cop.shape[1])), cop_train, epochs=100, batch_size=32, verbose=0)
cop_models['SSO-LSTM'] = cop_sso_lstm

print('  SSO-GRNN (Proposed) for COP...')
cop_sso_grnn = GRNN(sigma=best_sigma)
cop_sso_grnn.fit(X_train_cop, cop_train)
cop_models['SSO-GRNN (Proposed)'] = cop_sso_grnn

print()
print('Generating COP predictions...')
for name, model in cop_models.items():
    print(f'  Predicting COP with {name}...')
    if name in ['LSTM', 'SSO-LSTM']:
        pred_scaled = model.predict(X_test_cop.reshape((-1, 1, X_test_cop.shape[1])), verbose=0).flatten()
    else:
        pred_scaled = model.predict(X_test_cop)
    cop_predictions[name] = scaler_cop.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

cop_test_orig = scaler_cop.inverse_transform(cop_test.reshape(-1, 1)).flatten()

print()
print(f'{"Model":<25} {"MAE":<10} {"RMSE":<10} {"MAPE (%)":<12} {"R²":<10}')
print('-' * 67)

for name in cop_models.keys():
    cop_pred = cop_predictions[name]
    mae = mean_absolute_error(cop_test_orig, cop_pred)
    rmse = np.sqrt(mean_squared_error(cop_test_orig, cop_pred))
    mape = compute_mape(cop_test_orig, cop_pred)
    r2 = r2_score(cop_test_orig, cop_pred)
    print(f'{name:<25} {mae:<10.2f} {rmse:<10.2f} {mape:<12.2f} {r2:<10.4f}')

print()
print('=' * 80)
print('Defrosting and Non-Defrosting Period COP Performance (SSO-GRNN)')
print('=' * 80)

cop_pred_sso = cop_predictions['SSO-GRNN (Proposed)']
defrost_idx = defrost_test
non_defrost_idx = ~defrost_idx

if np.sum(defrost_idx) > 0:
    mae_def, rmse_def, mape_def, r2_def = compute_metrics(cop_test_orig[defrost_idx], cop_pred_sso[defrost_idx])
    print(f'Defrosting periods ({np.sum(defrost_idx)} samples):')
    print(f'  MAE: {mae_def:.2f}, RMSE: {rmse_def:.2f}, MAPE: {mape_def:.2f}%, R²: {r2_def:.4f}')

if np.sum(non_defrost_idx) > 0:
    mae_nondef, rmse_nondef, mape_nondef, r2_nondef = compute_metrics(cop_test_orig[non_defrost_idx],
                                                                      cop_pred_sso[non_defrost_idx])
    print(f'Non-defrosting periods ({np.sum(non_defrost_idx)} samples):')
    print(f'  MAE: {mae_nondef:.2f}, RMSE: {rmse_nondef:.2f}, MAPE: {mape_nondef:.2f}%, R²: {r2_nondef:.4f}')

print()
print('=' * 80)
print('Analysis Complete! Final Summary')
print('=' * 80)
print('Key Results - Heat Supply Prediction:')
for res in results:
    if res[0] == 'SSO-GRNN (Proposed)':
        print(f'  Proposed SSO-GRNN: MAPE = {res[4]:.2f}%, RMSE = {res[3]:.2f} kW, R² = {res[5]:.4f}')
print('Top 3 Features:')
for i, idx in enumerate(sorted_idx[:3]):
    print(f'  {i + 1}. {final_feature_names[idx]}: {sensitivity_indices[idx]:.4f}')
print('=' * 80)