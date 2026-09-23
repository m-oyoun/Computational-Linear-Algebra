import numpy as np

np.random.seed(42) 
X = 2 * np.random.rand(100, 1)  
y = 4 + 3 * X + np.random.randn(100, 1)  
N = len(X)
w = 0.0
b = 0.0
learning_rate = 0.05
epochs = 500  

print(" Vectorized Linear Regression Engine")
print("Training started")
print(f"Initial AI Guess > Weight (w): {w:.2f}, Bias (b): {b:.2f}\n")

for epoch in range(epochs):
    predictions = (X * w) + b
    errors = predictions - y
    mse = np.sum(errors ** 2) / N
    gradient_w = (2 / N) * np.sum(errors * X)
    gradient_b = (2 / N) * np.sum(errors)
    w = w - (learning_rate * gradient_w)
    b = b - (learning_rate * gradient_b)
    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | Total Error (MSE): {mse:5.2f} | Current w: {w:.2f}, b: {b:.2f}")

print("\nTraining Complete")
print(f"Final Weights Discovered by AI > Weight (w): {w:.4f} | Bias (b): {b:.4f}")
print("Target values were  > Weight (w): 3.0000 | Bias (b): 4.0000")

