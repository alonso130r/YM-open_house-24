import numpy as np
import plotly.graph_objects as go

# Define the Rastrigin function and its gradient
def f(x, y):
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

def grad_f(x, y):
    df_dx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    df_dy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)
    return np.array([df_dx, df_dy])

def stochastic_grad_f(x, y, noise_std=1.0):
    grad = grad_f(x, y)
    noise = np.random.normal(0, noise_std, size=grad.shape)
    return grad + noise

# SOPHIA optimizer implementation
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

# Parameters for the optimizer
optimizer = SophiaOptimizer(lr=0.01, beta1=0.9, beta2=0.99, rho=0.04, weight_decay=0.01)
max_iters = 100  # Increased number of iterations for better convergence
x0 = np.array([4.55, 4.50])  # Starting point

# Lists to store the path of the optimizer
x_vals = [x0[0]]
y_vals = [x0[1]]
z_vals = [f(x0[0], x0[1])]

# Run the optimization process
for i in range(max_iters):
    grads = stochastic_grad_f(x0[0], x0[1], noise_std=1.0)  # Stochastic gradient
    x0 = optimizer.update(x0, grads)  # Update the point using SOPHIA optimizer
    x_vals.append(x0[0])
    y_vals.append(x0[1])
    z_vals.append(f(x0[0], x0[1]))

# Create a meshgrid for plotting the function surface
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create the 3D surface plot
surface = go.Surface(
    x=X, y=Y, z=Z, colorscale='Viridis', showscale=False, opacity=0.9, name='Function Surface'
)

# Create initial scatter point
scatter = go.Scatter3d(
    x=[x_vals[0]],
    y=[y_vals[0]],
    z=[z_vals[0]],
    mode='markers',
    marker=dict(color='red', size=5),
    name='Optimization Path'
)

# Create frames for the animation
frames = []
for i in range(1, len(x_vals), 5):  # Adjusted step size for fewer frames
    frames.append(go.Frame(
        data=[
            # Only update the data for the optimization path
            go.Scatter3d(
                x=x_vals[:i+1],
                y=y_vals[:i+1],
                z=z_vals[:i+1],
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(color='red', size=3),
            )
        ],
        name=str(i),
        traces=[1]  # This frame will update trace with index 1 (scatter plot)
    ))

# Set up the layout with buttons and sliders
layout = go.Layout(
    title='Optimization of the Rastrigin Function using SOPHIA Optimizer with SGD',
    scene=dict(
        xaxis=dict(title='X', range=[-5.12, 5.12]),
        yaxis=dict(title='Y', range=[-5.12, 5.12]),
        zaxis=dict(title='f(X, Y)', range=[0, 80]),
        aspectratio=dict(x=1, y=1, z=0.7)
    ),
    updatemenus=[
        dict(
            type='buttons',
            showactive=False,
            buttons=[
                dict(
                    label='Play',
                    method='animate',
                    args=[
                        None,
                        dict(
                            frame=dict(duration=50, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode='immediate'
                        )
                    ]
                )
            ],
            x=0.1,
            y=0,
            xanchor='right',
            yanchor='top'
        )
    ],
    sliders=[
        dict(
            steps=[
                dict(
                    method='animate',
                    args=[
                        [str(k)],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                            mode='immediate'
                        )
                    ],
                    label=str(k)
                ) for k in range(0, len(frames))
            ],
            active=0,
            x=0.1,
            y=0,
            currentvalue=dict(prefix='Iteration: ', visible=True),
            len=0.9
        )
    ]
)

# Create the figure and add the surface and initial scatter plot
fig = go.Figure(data=[surface, scatter], layout=layout, frames=frames)

# Display the figure
fig.show()
