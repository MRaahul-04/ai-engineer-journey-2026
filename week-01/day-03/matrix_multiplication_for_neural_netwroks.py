# The rule: (rows_A, cols_A) @ (rows_B, cols_B) only works if cols_A == rows_B. The output shape is (rows_A,
# cols_B) — the matching middle numbers "cancel out."


# looping way to do the matrix multiplication
A = [[1,2,3],[4,5,6]]
B = [[7,8],[9,10],[11,12]]

def matmul_scratch(A, B):
    rows_A = len(A)          # how many rows in A
    cols_A = len(A[0])       # how many columns in A
    cols_B = len(B[0])       # how many columns in B

    C = []                          # will hold the final result matrix
    for i in range(rows_A):         # pick a row of A
        row_result = []              # this row's finished numbers go here
        for j in range(cols_B):      # pick a column of B
            total = 0                 # fresh running total for THIS (i, j) pair
            for k in range(cols_A):
                total += A[i][k] * B[k][j]   # walk position-by-position, multiply, accumulate
            row_result.append(total)   # k-loop is done — total is finished, save it
        C.append(row_result)         # this row is done — save it into C
    return C

print("Looping way:", matmul_scratch(A, B))
print("\n")

# NumPy replaces all three of those loops with one symbol:

import numpy as np
C = np.array(A) @ np.array(B)
print("1. Numpy way using @:\n", C)

# Example 2
A = [[3, 1],
     [2, 4]]      #shape (2, 2)

B = [[5, 0, 2],
     [1, 3, 4]]    #shape (2, 3)

C = np.array(A) @ np.array(B)
print("\n 2. Numpy way using @:\n", C)

"""
# For the dot product part: yes, @ works. When you use @ on two 1-D vectors (not 2-D matrices), NumPy is smart enough
# to know you mean the plain dot product — it's a special case of matmul where each "row" and "column" is just the 
# whole vector.
#
# python
# A = np.array([9, 2, 1])
# B = np.array([8, 3, 2])
#
# A @ B          # → 80
# np.dot(A, B)   # → 80   (identical result)
#
# They're genuinely interchangeable for vectors — @ and np.dot compute the exact same number here. np.dot is actually
# the more common convention for vector-vector work; @ tends to be reserved for matrix-vs-matrix, just by style 
# convention, not a hard rule.
#
# But norm and cosine similarity are a different story. @ (matrix multiplication) only ever does one thing: 
# multiply-matching-positions-and-sum. It has no concept of "square root" or "divide." So:
#
# Norm needs np.linalg.norm(A) — a completely separate function, because it's square-sum-square root, 
# not a multiplication between two things. Cosine similarity needs the division step after the dot product: (A @ B) /
# (norm_A * norm_B). The @ only ever gives you the numerator — you still have to build the rest by hand.
#
# So the honest picture: @ replaces the dot product step, but it can't absorb norm or the final division — those are 
# separate operations that dot product feeds into, not something @ can do on its own.
"""

# Fake neural network layer
"""
A neural network "layer" takes an input, multiplies it by a 
weights matrix, adds a bias, and (usually) squashes the result through an activation function

output = (input @ weights) + bias

The simplest one, ReLU (Rectified Linear Unit), is almost insultingly simple: if the number is negative, 
make it 0. If it's positive, leave it alone.
"""
# Positive arrays
import numpy as np

x = np.array([2, 3])        # shape (1, 2)  — 1 row, 2 features
W = np.array([[1, 0, 2],
     [0, 1, 1]])   # shape (2, 3)  — 2 in, 3 out

b = np.array([1, 1, 0])     # shape (3,)

def relu(x):
    return np.maximum(0, x)  # # NumPy's version of your max(0, x), works on whole arrays at once

z = (x @ W) + b        # linear step: matmul + bias
output = relu(z)     # activation step

print("\n z (pre-activation):", z)
print("output (post-ReLU):", output)

# Negative b array
import numpy as np

x = np.array([2, 3])        # shape (1, 2)  — 1 row, 2 features
W = np.array([[1, 0, 2],
     [0, 1, 1]])   # shape (2, 3)  — 2 in, 3 out

b = np.array([1, -5, -2])     # shape (3,)

def relu(x):
    return np.maximum(0, x)  # # NumPy's version of your max(0, x), works on whole arrays at once

z = (x @ W) + b        # linear step: matmul + bias
output = relu(z)     # activation step

print("\n z (pre-activation):", z)
print("output (post-ReLU):", output)
