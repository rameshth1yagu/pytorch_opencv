import torch  # PyTorch for tensor operations
import torchviz  # For visualizing computation graphs
import matplotlib.pyplot as plt  # For plotting
from torchviz import make_dot  # For graph visualization

# Example 1: Simple autograd and backpropagation loop
def autograd_backpropagation_example_1():
    w = torch.tensor(10.0, requires_grad=True)  # Initialize weight with gradient tracking
    target = torch.tensor(4.0)  # Target value

    for i in range(25):  # Run for 25 epochs
        prediction = w  # Model prediction (identity)
        loss = (prediction - target) ** 2  # Mean squared error loss
        loss.backward()  # Compute gradients
        print(f"Epoch: {i}, prediction: {prediction.item():.2f}, loss: {loss.item():.2f}, grad: {w.grad.item():.2f}")
        with torch.no_grad():  # Update weights without tracking
            w -= w.grad * 0.1  # Gradient descent step
            w.grad.zero_()  # Reset gradients for next iteration

# Example 2: Manual derivative calculation and autograd
def derivative_example_1():
    x = torch.tensor([2.0], requires_grad=True)  # Variable x
    y = torch.tensor([3.0], requires_grad=True)  # Variable y
    loss = x * y + y ** 2  # Function to differentiate
    loss.backward()  # Compute gradients
    print(f"x.grid: {x.grad.item():.2f}") # Gradient of x (should be y)
    print(f"y.grid: {y.grad.item():.2f}") # Gradient of y (should be x + 2y)

# Example 3: Vectorized derivative and autograd
def derivative_example_2():
    x = torch.tensor([2.0, 5.0], requires_grad=True)  # Vector x
    y = torch.tensor([3.0, 7.0], requires_grad=True)  # Vector y
    z = x * y + y ** 2  # Vectorized function
    print(f"Value of z: {z}")
    z.retain_grad() # Retain grad for z (for visualization)
    z.sum().backward()  # Backpropagate through sum
    z_detached = z.detach() # Detach z from computation graph
    print(f"x.grid: {x.grad}") # Gradient of x (should be y)
    print(f"y.grid: {y.grad}") # Gradient of y (should be x + 2y)
    print(f"z.grid: {z.grad}") # Gradient of z
    print(f"Result of the operation: {z_detached}")
    print(f"is z tracking turn off: {z.requires_grad}")
    print(f"is z_detached tracking turn off: {z_detached.requires_grad}")
    print(f"is z tracking turn off: {z.requires_grad}")
    return x, y, z

# Visualize the computation graph using torchviz
def visualize(x, y, z):
    dot = make_dot(z, params={'x': x, 'y': y, 'z': z})  # Create graph
    dot.render('output', format='png')  # Save graph as PNG

# Uncomment to run examples
#autograd_backpropagation_example_1()
#derivative_example_1()
x, y, z = derivative_example_2()  # Run vectorized example
visualize(x, y, z)  # Visualize computation graph

