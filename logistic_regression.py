import numpy as np  
# STEP 1: MATHEMATICAL MECHANICS (THE MECHANICAL SWITCH & LOSS TRACKER)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    N = len(y_true)  
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = - (1 / N) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return loss  
print("--- Step 1: Subroutines Loaded Successfully ---")
print(f"Testing Sigmoid at z=+10: {sigmoid(10):.4f} (Expect near 1.0)")
print(f"Testing Sigmoid at z=-10: {sigmoid(-10):.4f} (Expect near 0.0)")
print(f"Testing Sigmoid at z=0:   {sigmoid(0):.4f} (Expect exactly 0.5)")

#  THE PATIENT DATA MATRIX (GENERATING THE CASE FILES)


np.random.seed(42)  
X = 2 * np.random.rand(100, 1)  
y = (X > 1.0).astype(float)  
w = 0.0
b = 0.0

print("\n--- Step 2: Patient Dataset Generated ---")
print(f"Total data entries loaded: {len(X)}")
print(f"Sample Patient 1 -> Tumor Size: {X[0][0]:.2f} cm | Real Label: {int(y[0][0])}")
print(f"Sample Patient 2 -> Tumor Size: {X[1][0]:.2f} cm | Real Label: {int(y[1][0])}")


# STEP 3: THE GRADIENT DESCENT OPTIMIZATION LOOP


learning_rate = 0.1  
epochs = 301        
N = len(X)           

print("\n--- Step 3: Commencing Gradient Descent Optimizer ---")
for epoch in range(epochs):
    linear_score = (X * w) + b
    predictions = sigmoid(linear_score)  
    loss = compute_loss(y, predictions)
    errors = predictions - y
    gradient_w = (1 / N) * np.sum(errors * X)
    gradient_b = (1 / N) * np.sum(errors)
    w = w - (learning_rate * gradient_w)
    b = b - (learning_rate * gradient_b)
    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | Log Loss: {loss:.4f} | Current Parameters -> w: {w:.2f}, b: {b:.2f}")

print("\n--- Training Pipeline Complete ---")
print(f"Final Optimized Weight (w): {w:.4f} (Controls curve steepness)")
print(f"Final Optimized Intercept Bias (b): {b:.4f} (Controls curve position shift)")