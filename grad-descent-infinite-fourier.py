import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm
import tkinter as tk
import time

# Global variables
stop_flag = False
waiting = False
iteration = 0
max_iters = 100
wait_time = 10  # seconds
start_time = None
optimizer = None
x0 = None
x_vals = []
y_vals = []
z_vals = []
A_n = B_n = k_nx = k_ny = phi_n = None
f = None
grad_f = None

def generate_random_fourier_function(N_terms=10):
    # Generate random amplitudes, frequencies, and phases
    A_n = np.random.uniform(-1, 1, N_terms)
    B_n = np.random.uniform(-1, 1, N_terms)
    k_nx = np.random.uniform(1, 3, N_terms)
    k_ny = np.random.uniform(1, 3, N_terms)
    phi_n = np.random.uniform(0, 2 * np.pi, N_terms)
    return A_n, B_n, k_nx, k_ny, phi_n

def initialize_function():
    global A_n, B_n, k_nx, k_ny, phi_n, f, grad_f
    A_n, B_n, k_nx, k_ny, phi_n = generate_random_fourier_function(N_terms=5)
    def f_local(x, y):
        val = np.zeros_like(x)
        for i in range(len(A_n)):
            val += A_n[i] * np.cos(k_nx[i] * x + k_ny[i] * y + phi_n[i])
            val += B_n[i] * np.sin(k_nx[i] * x + k_ny[i] * y + phi_n[i])
        return val
    def grad_f_local(x, y):
        grad_x = np.zeros_like(x)
        grad_y = np.zeros_like(y)
        for i in range(len(A_n)):
            angle = k_nx[i] * x + k_ny[i] * y + phi_n[i]
            grad_x += -A_n[i] * k_nx[i] * np.sin(angle) + B_n[i] * k_nx[i] * np.cos(angle)
            grad_y += -A_n[i] * k_ny[i] * np.sin(angle) + B_n[i] * k_ny[i] * np.cos(angle)
        return np.array([grad_x, grad_y])
    f = f_local
    grad_f = grad_f_local

def initialize_optimizer():
    global optimizer, x0, x_vals, y_vals, z_vals, iteration
    iteration = 0

    # Perform a grid search to find the maximum
    x_search = y_search = np.linspace(-5, 5, 200)
    X_search, Y_search = np.meshgrid(x_search, y_search)
    Z_search = f(X_search, Y_search)

    # Find the index of the maximum value
    max_index = np.unravel_index(np.argmax(Z_search), Z_search.shape)
    x_max = X_search[max_index]
    y_max = Y_search[max_index]

    # Initialize starting point at the maximum
    x0 = np.array([x_max, y_max])
    x_vals = [x0[0]]
    y_vals = [x0[1]]
    z_vals = [f(x0[0], x0[1])]

    class SophiaOptimizer:
        def __init__(self, lr=0.01, beta1=0.9, beta2=0.99, rho=0.04, epsilon=1e-8, weight_decay=0.01):
            self.lr = lr              # Base learning rate
            self.beta1 = beta1        # Exponential decay rate for first moment estimates
            self.beta2 = beta2        # Exponential decay rate for second moment (Hessian) estimates
            self.rho = rho            # Damping factor for Hessian approximation
            self.epsilon = epsilon    # Small constant to prevent division by zero
            self.weight_decay = weight_decay  # Weight decay coefficient
            self.m = np.zeros(2)      # First moment vector (momentum)
            self.h = np.zeros(2)      # Second moment vector (approximate Hessian diagonal)
            self.t = 0                # Time step

        def update(self, params, grads):
            self.t += 1

            # Apply weight decay directly to the parameters
            params = params * (1 - self.lr * self.weight_decay)

            # Update first moment estimate (momentum)
            self.m = self.beta1 * self.m + (1 - self.beta1) * grads

            # Approximate Hessian diagonal (second moment estimate)
            self.h = self.beta2 * self.h + (1 - self.beta2) * (grads ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m / (1 - self.beta1 ** self.t)

            # Compute adjusted learning rate for each parameter
            lr_t = self.lr / (np.sqrt(self.h) + self.rho + self.epsilon)

            # Update parameters
            params = params - lr_t * m_hat

            return params

    optimizer = SophiaOptimizer(lr=0.01, beta1=0.9, beta2=0.99, rho=0.04, weight_decay=0.01)

def plot_function():
    global X, Y, Z, ax, canvas, path_line
    x_range = y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    ax.clear()
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(Z.min(), Z.max())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title('Optimization of Random Fourier Function using SOPHIA Optimizer')

    # Plot the initial point
    path_line, = ax.plot(x_vals, y_vals, z_vals, color='red', marker='o', markersize=5)

    canvas.draw()

def optimization_step():
    global stop_flag, waiting, iteration, x0, x_vals, y_vals, z_vals, start_time

    if stop_flag:
        root.quit()
        return

    if not waiting:
        if iteration < max_iters:
            # Perform optimization step
            noise_std = 1.0
            grads = grad_f(x0[0], x0[1]) + np.random.normal(0, noise_std, size=2)
            x_new = optimizer.update(x0, grads)
            x_vals.append(x_new[0])
            y_vals.append(x_new[1])
            z_vals.append(f(x_new[0], x_new[1]))
            x0 = x_new
            iteration += 1

            # Update the path on the plot
            path_line.set_data(x_vals, y_vals)
            path_line.set_3d_properties(z_vals)

            canvas.draw()
            root.after(1, optimization_step)  # Schedule next step
        else:
            # Start waiting
            waiting = True
            start_time = time.time()
            root.after(100, optimization_step)
    else:
        # Waiting period
        elapsed_time = time.time() - start_time
        if elapsed_time >= wait_time:
            # Reset and start over
            waiting = False
            initialize_function()
            initialize_optimizer()
            plot_function()
            root.after(1, optimization_step)
        else:
            root.after(100, optimization_step)

def stop():
    global stop_flag
    stop_flag = True

# Create a Tkinter window
root = tk.Tk()
root.title("Optimization with SOPHIA Optimizer")

# Create the matplotlib figure and axes
fig = plt.Figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Create a canvas to embed the figure in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Add the stop button
stop_button = tk.Button(root, text="Stop", command=stop)
stop_button.pack()

# Initialize the function, optimizer, and plot
initialize_function()
initialize_optimizer()
plot_function()

# Start the optimization loop
root.after(1, optimization_step)

# Start the Tkinter event loop
root.mainloop()
