import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm
import tkinter as tk
import time

stop_flag = False
waiting = False
iteration = 0
max_iters = 250
wait_time = 2  # seconds
start_time = None
optimizer = None
x0 = None
x_vals = []
y_vals = []
z_vals = []
centers = amplitudes = widths = None
f = None
grad_f = None

def generate_random_gaussian_function(N_terms=30):
    # generate random centers, amplitudes, and widths for the Gaussians
    centers = np.random.uniform(-5, 5, (N_terms, 2))
    amplitudes = np.random.uniform(-1, 1, N_terms)
    widths = np.random.uniform(0.5, 1.5, N_terms)
    return centers, amplitudes, widths

def initialize_function():
    global centers, amplitudes, widths, f, grad_f
    centers, amplitudes, widths = generate_random_gaussian_function(N_terms=30)
    
    def f_local(x, y):
        val = 0
        for i in range(len(amplitudes)):
            dx = x - centers[i, 0]
            dy = y - centers[i, 1]
            exponent = -((dx**2 + dy**2) / (2 * widths[i]**2))
            val += amplitudes[i] * np.exp(exponent)
        return val
    
    def grad_f_local(x, y):
        grad_x = 0
        grad_y = 0
        for i in range(len(amplitudes)):
            dx = x - centers[i, 0]
            dy = y - centers[i, 1]
            exponent = -((dx**2 + dy**2) / (2 * widths[i]**2))
            common_factor = amplitudes[i] * np.exp(exponent) / widths[i]**2
            grad_x += -common_factor * dx  
            grad_y += -common_factor * dy  
        return np.array([grad_x, grad_y])

    
    f = f_local
    grad_f = grad_f_local

def initialize_optimizer():
    global optimizer, x0, x_vals, y_vals, z_vals, iteration
    iteration = 0

    # perform a grid search to find the maximum
    x_search = y_search = np.linspace(-5, 5, 200)
    X_search, Y_search = np.meshgrid(x_search, y_search)
    Z_search = f(X_search, Y_search)

    # find the index of the maximum value
    max_index = np.unravel_index(np.argmax(Z_search), Z_search.shape)
    x_max = X_search[max_index]
    y_max = Y_search[max_index]

    # add a small perturbation to avoid zero gradient
    perturbation = np.random.uniform(-1.0, 1.0, size=2)
    x0 = np.array([x_max, y_max]) + perturbation

    # clip x0 to stay within bounds
    x0 = np.clip(x0, -5, 5)

    x_vals = [x0[0]]
    y_vals = [x0[1]]
    z_vals = [f(x0[0], x0[1])]

    class AdamWOptimizer:
        def __init__(self, lr=0.1, beta1=0.95, beta2=0.9, epsilon=1e-8, weight_decay=0.01):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.weight_decay = weight_decay
            self.m = np.zeros(2)  # first moment vector
            self.v = np.zeros(2)  # second moment vector
            self.t = 0            

        def update(self, params, grads):
            self.t += 1

            # apply weight decay directly to the parameters
            params = params * (1 - self.lr * self.weight_decay)

            # update biased first moment estimate
            self.m = self.beta1 * self.m + (1 - self.beta1) * grads

            # update biased second raw moment estimate
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

            # compute bias-corrected first moment estimate
            m_hat = self.m / (1 - self.beta1 ** self.t)

            # compute bias-corrected second raw moment estimate
            v_hat = self.v / (1 - self.beta2 ** self.t)

            # update parameters to minimize the loss function
            params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            return params

    optimizer = AdamWOptimizer(lr=0.1, beta1=0.95, beta2=0.9, weight_decay=0.01)

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
    ax.set_title('Optimization of Random Gaussian Function using AdamW Optimizer')

    path_line, = ax.plot(x_vals, y_vals, z_vals, color='red', marker='o', markersize=5)

    canvas.draw()

def optimization_step():
    global stop_flag, waiting, iteration, x0, x_vals, y_vals, z_vals, start_time

    if stop_flag:
        root.quit()
        return

    if not waiting:
        if iteration < max_iters:
            # perform optimization step
            noise_std = 0.1  # small noise level
            grads = grad_f(x0[0], x0[1]) + np.random.normal(0, noise_std, size=2)

            x_new = optimizer.update(x0, grads)
            # clip x_new to the bounds [-5, 5]
            x_new = np.clip(x_new, -5, 5)
            x_vals.append(x_new[0])
            y_vals.append(x_new[1])
            z_vals.append(f(x_new[0], x_new[1]))
            x0 = x_new
            iteration += 1

            # update the path on the plot
            path_line.set_data(x_vals, y_vals)
            path_line.set_3d_properties(z_vals)

            canvas.draw()
            root.after(1, optimization_step)  # schedule next step
        else:
            waiting = True
            start_time = time.time()
            root.after(100, optimization_step)
    else:
        elapsed_time = time.time() - start_time
        if elapsed_time >= wait_time:
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

# create a tkinter window
root = tk.Tk()
root.title("Optimization with AdamW Optimizer")

# create the matplotlib figure and axes
fig = plt.Figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# create a canvas to embed the figure in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# add the stop button
stop_button = tk.Button(root, text="Stop", command=stop)
stop_button.pack()

# tnitialize the function, optimizer, and plot
initialize_function()
initialize_optimizer()
plot_function()

root.after(1, optimization_step)

root.mainloop()
