import numpy as np

def gaussian_elimination_general(A, b):
    """
    Solves an N x N system of equations dynamically using loops.
    Handles any size system of linear equations.
    """
    n = len(b)
    
    # Combine A and b into a single augmented matrix as floats
    M = np.hstack((A.astype(float), b.astype(float).reshape(-1, 1)))
    for i in range(n):
        for j in range(i + 1, n):
            factor = M[j][i] / M[i][i]
            M[j] = M[j] - factor * M[i]
            

    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        total = M[i][-1]
        
        for j in range(i + 1, n):
            total -= M[i][j] * x[j]

        x[i] = total / M[i][i]
        
    return x

# --- Testing the N x N Solver with a 3x3 system ---
# 2x + 3y + z = 11
# 1x - 2y + 4z = 11
# 3x + 1y - 2z = -1
matrix_A = np.array([[2, 3, 1],
                     [1, -2, 4],
                     [3, 1, -2]])

vector_b = np.array([11, 11, -1])

answers = gaussian_elimination_general(matrix_A, vector_b)

print("\n--- N x N Dynamic Matrix Solver ---")
print(f"Calculated Solutions [x, y, z]: [{answers[0]:.1f}, {answers[1]:.1f}, {answers[2]:.1f}]")