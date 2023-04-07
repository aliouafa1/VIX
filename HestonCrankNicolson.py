import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

# Parameters
S0 = 100
r = 0.02
V0 = 0.04
kappa = 1
theta = 0.04
sigma = 0.5
rho = -0.5
T = 1
K = 100

# Grid
N = 2**8
xmax = 1
alpha = 10
x, dx = np.linspace(-xmax, xmax, N, retstep=True)
k = fft.fftfreq(N, d=dx) * 2*np.pi
k2 = k**2

# Set the Initial condition
v = np.maximum(V0, np.zeros(N))

# Time stepping
dt = 1/12
nsteps = int(T/dt)
for i in range(nsteps):
    # Fourier transform of volatility
    v_hat = fft.fft(v)
    
    # Set Heston PDE in Fourier space (system of linear equations that can be efficiently solved using the FFT algorithm)
    a = kappa*theta / sigma**2 - 0.5
    b = rho*kappa / (sigma*dx) - 0.5
    c = k**2 / dx**2 + rho*kappa / (sigma*dx)
    A = 1 + dt*(0.25*b**2*v_hat - 0.5*a*b*1j*k*v_hat - 0.5*c*v_hat)
    B = dt*(-0.5*b**2*v_hat + (1j*k*r - 0.5*a*sigma**2)*b*v_hat + 0.5*sigma**2*c*v_hat)
    C = dt*(0.25*b**2*v_hat + 0.5*a*b*1j*k*v_hat - 0.5*c*v_hat)
    
    # We set Inverse Fourier transform
    v = np.real(fft.ifft((B + np.sqrt(B**2 - 4*A*C)) / (2*A)))
    
    # then we can Enforce boundary condition
    v[0] = V0
    v[-1] = V0

# Now we can Calculate variance swap fair value
v_mean = np.mean(v)
varswap = np.exp(-r*T) * (v_mean - V0)
print(f"Variance Swap Fair Value: {varswap:.4f}")
