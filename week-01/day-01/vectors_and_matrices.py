import numpy as np

# ── Vector: one house, described by numbers ──────────────────
house_1 = np.array([1200, 3, 5, 85])
# [size_sqft, bedrooms, age_years, price_lakh]

print("This is a vector:", house_1)
print("Shape:", house_1.shape) #That comma matters: it tells you this is a 1-D array (a vector), not a row or column of a bigger structure yet.
print("Data type:", house_1.dtype)

print("\n")
# ── Matrix: many houses, stacked ──────────────────────────────
houses = np.array([
    [1200, 3, 5, 85],   # house 1
    [800, 2, 10, 55],   # house 2
    [2000, 4, 2, 140],  # house 3
    [950, 2, 8, 62],    # house 4
    [1500, 5, 6, 54],   # house 5
])

print("This is a matrix:", houses)
print("Shape:", houses.shape)  # (4, 4) → 4 houses, 4 features each

print("\n")

# Grab one house (one row = one vector, back out of the matrix)
print(houses[0])          # first house entirely

# Grab one feature across ALL houses (one column)
print(houses[:, 0])       # size_sqft for every house — this is the "all sizes" vector

# Grab a single value
print(houses[1, 2])       # house 2's age (row 1, column 2)

# Test
print(houses[:, 3])       # Grabs the prices of all houses

# The key rule: indexing with a single integer removes a dimension. Slicing (even a slice of size 1) keeps it.
print(houses[0])      # shape (4,)   — integer index → NumPy says "give me exactly row 0, and drop the row dimension"
print(houses[0:1])    # shape (1,4)  — still "sliced," so it stays a matrix. slice → Give me houses from index 0 up to (not including) 1" → you get a container holding one house. Still a matrix, just with 1 row. Shape (1, 4)
print(houses[0:2])

print(houses[0].shape)
print(houses[0:1].shape)
print(houses[0] == houses[0:1])
print((houses[0] == houses[0:1]).all())   # True — "are ALL elements equal?"

print("\n")
# Say you want price per square foot for every house — a completely normal thing to want in real ML data prep. The naive, non-ML way (a Python loop):

# The "slow" way - looping like a beginner
price_per_sqft = []
for house in houses:
    price_per_sqft.append(house[3] / house[0])
print("Slow loop:", price_per_sqft)

# Now the vectorized way — the linear algebra way:
price_per_sqft = houses[:, 3] / houses[:, 0]
print("Fast loop:",price_per_sqft)

# One line. No loop. Run both versions and confirm you get the identical numbers. Here's what's actually happening
# underneath, and it's worth sitting with: houses[:, 3] grabs the "all prices" vector, houses[:, 0] grabs the "all
# sizes" vector, and the / divides them element-wise — position 0 divided by position 0, position 1 by position 1,
# and so on, across the entire vector, simultaneously. Not a loop in disguise — an actual single operation applied
# across all 4 (or 4 million) values at once, handled by optimized C code under the hood instead of slow Python
# iteration.


# Homework 1 (Add a 5th house to `houses`, then compute price_per_sqft for all 5 — vectorized only, no loop)
price_per_sqft = houses[:, 3] / houses[:, 0]
print("vectorized compute for all 5:", price_per_sqft)

# Homework 1 (Write one sentence in your own words: "why does a Python loop become a real problem in ML?") Ans -
"""
> My ANS 
- Python Loop is real problem in python because it goes through each row one by one to process the operation via
loop, which is actually fine for the small number for dataset or houses as we took example of. But ML actually
operates on the large datasets which can't be done via looping one by one row. there comes the vectorized way where
one single line code process all the rows at once as per the provided logic and gives the results accordingly.
That's why vectorized way is recommended for the ML operations.

> Refined way 
- Python loops are slow because each iteration pays Python's interpreter 
overhead individually — vectorization skips that by pushing the whole operation into optimized, 
compiled code that processes all elements in one call.

> Interview Way 
- the precise technical reason isn't just "loops are slow to write" — it's that Python loops run one 
instruction at a time in the Python interpreter itself, while vectorized NumPy operations push the entire loop down
into pre-compiled C code that runs all at once, often using CPU-level parallelism (SIMD instructions). So it's not 
just "fewer lines" — it's fundamentally faster execution, often 10-100x, because you've skipped Python's 
interpreter overhead entirely for each element.
"""


