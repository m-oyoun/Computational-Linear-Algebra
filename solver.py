import numpy as np

def solve_2x2_system(A, b):
    """
    Solves a 2x2 system of equations using forward elimination 
    and back substitution (The foundation of Gaussian Elimination).
    """
    A = A.astype(float)
    b = b.astype(float)

    factor = A[1][0] / A[0][0]
    A[1] = A[1] - factor * A[0]
    b[1] = b[1] - factor * b[0]
    y = b[1] / A[1][1]
    x = (b[0] - A[0][1] * y) / A[0][0]
    
    return np.array([x, y])

# --- Testing the 2x2 Solver ---
# Equations: 2x + 3y = 8 and 1x - 2y = -3
matrix_A = np.array([[2, 3], 
                     [1, -2]])
vector_b = np.array([8, -3])

my_solutions = solve_2x2_system(matrix_A, vector_b)

print("Custom 2x2 Gaussian Elimination")
print(f"Calculated x: {my_solutions[0]:.1f}")
print(f"Calculated y: {my_solutions[1]:.1f}")