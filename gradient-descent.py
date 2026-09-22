import numpy as np
def get_derivative(w):
    return 2 * (w - 5)

current_w = 0.0          
learning_rate = 0.1      
iterations = 20          

print("--- Gradient Descent Simulation ---")
print(f"Starting Weight: {current_w}\n")

for step in range(iterations):
    slope = get_derivative(current_w)
    
    current_error = (current_w - 5) ** 2
    current_w = current_w - (learning_rate * slope)

    if step % 2 == 0:
        print(f"Step {step:02d} | Current Error: {current_error:5.2f} | Updated Weight: {current_w:.4f}")

print(f"\nFinal Optimized Weight found by AI: {current_w:.4f} (Target was 5.0)")
