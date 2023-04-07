import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define the neural network
def create_nn():
    nn = keras.Sequential([
        keras.layers.Dense(units=32, input_shape=(2,), activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=1, activation=None)
    ])
    nn.compile(optimizer='adam', loss='mse')
    return nn

# Generate training data for VarSwap
def generate_varswap_data():
    np.random.seed(42)
    K = np.arange(0.8, 1.21, 0.01)
    T = np.arange(0.25, 2.01, 0.25)
    X = np.array(np.meshgrid(K, T)).T.reshape(-1, 2)
    y = np.zeros_like(X[:, 0])
    for i in range(len(y)):
        y[i] = heston_varswap_price(X[i, 0], X[i, 1], alpha, rho, kappa, theta, sigma)
    return X, y

# Generate training data for VolSwap
def generate_volswap_data():
    np.random.seed(42)
    K = np.arange(0.8, 1.21, 0.01)
    T = np.arange(0.25, 2.01, 0.25)
    X = np.array(np.meshgrid(K, T)).T.reshape(-1, 2)
    y = np.zeros_like(X[:, 0])
    for i in range(len(y)):
        y[i] = heston_volswap_price(X[i, 0], X[i, 1], alpha, rho, kappa, theta, sigma)
    return X, y

# Train the neural network for VarSwap
def train_varswap_nn(nn, X_train, y_train):
    nn.fit(X_train, y_train, epochs=1000, verbose=0)
    
# Train the neural network for VolSwap
def train_volswap_nn(nn, X_train, y_train):
    nn.fit(X_train, y_train, epochs=1000, verbose=0)
    
# Compute the fair value of VarSwap using the neural network
def nn_varswap_price(K, T):
    return nn_varswap.predict(np.array([[K, T]]))[0, 0]

# Compute the fair value of VolSwap using the neural network
def nn_volswap_price(K, T):
    return nn_volswap.predict(np.array([[K, T]]))[0, 0]

# Generate test data for VarSwap
X_test_var, y_test_var = generate_varswap_data()

# Train the neural network for VarSwap
nn_varswap = create_nn()
train_varswap_nn(nn_varswap, X_train_var, y_train_var)

# Compute the fair value of VarSwap for test data using the neural network
y_pred_var = nn_varswap.predict(X_test_var)

# Calculate the root mean squared error of the neural network for VarSwap
mse_var = np.mean((y_pred_var.flatten() - y_test_var) ** 2)
rmse_var = np.sqrt(mse_var)
print('Root Mean Squared Error (VarSwap Neural Network): {:.4f}'.format(rmse_var))

# Generate test data for VolSwap
X_test_vol, y_test_vol = generate_volswap_data()

# Train the neural network for VolSwap
nn_volswap = create_nn()
train_volswap_nn(nn_volswap, X_train_vol, y_train_vol)


# Define the neural network model
def create_model(input_shape):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the neural network on the VarSwap and VolSwap fair values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model(input_shape=X_train.shape[1:])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=2)

# Calculate the neural network predicted fair values and compare with the Monte Carlo results
y_pred = model.predict(X_test)
var_swap_nn = y_pred[:, 0]
vol_swap_nn = y_pred[:, 1]

print("VarSwap NN Fair Value: ", np.mean(var_swap_nn))
print("VolSwap NN Fair Value: ", np.mean(vol_swap_nn))

# Calculate the convergence rate of the neural network approach
var_swap_nn_convergence = np.zeros(len(sample_sizes))
vol_swap_nn_convergence = np.zeros(len(sample_sizes))

for i, sample_size in enumerate(sample_sizes):
    X_train_subsample = X_train[:sample_size]
    y_train_subsample = y_train[:sample_size]
    model.fit(X_train_subsample, y_train_subsample, epochs=10, batch_size=64, verbose=0)
    y_pred_subsample = model.predict(X_test)
    var_swap_nn_convergence[i] = np.abs(np.mean(y_pred_subsample[:, 0]) - np.mean(var_swap_mc)) / np.std(var_swap_mc)
    vol_swap_nn_convergence[i] = np.abs(np.mean(y_pred_subsample[:, 1]) - np.mean(vol_swap_mc)) / np.std(vol_swap_mc)

# Plot the convergence rates
plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, var_swap_mc_convergence, label='VarSwap MC')
plt.plot(sample_sizes, vol_swap_mc_convergence, label='VolSwap MC')
plt.plot(sample_sizes, var_swap_nn_convergence, label='VarSwap NN')
plt.plot(sample_sizes, vol_swap_nn_convergence, label='VolSwap NN')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Sample Size')
plt.ylabel('Convergence Rate')
plt.title('Convergence Rates of Monte Carlo and Neural Network Approaches')
plt.legend()
plt.show()
