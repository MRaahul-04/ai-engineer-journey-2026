
"""
You've got two vectors. Say these represent two users on a movie platform, based on how much they rated Action,
Comedy, and Romance (scale 0–10):
"""
from nbformat.v4 import output_from_msg

user_A = [9, 2, 1]   # loves action, hates the rest
user_B = [8, 3, 2]   # also loves action, hates the rest
user_C = [1, 2, 9]   # loves romance, hates the rest
user_D = [90, 20, 10]   # same taste as user_A, just... rates everything 10x more intensely

"""
Think about what you'd naturally do if I gave you these two lists and asked "how much do they agree":
```
user_A = [9, 2, 1]
user_B = [8, 3, 2]
```
What if you multiplied matching positions together, then added those up?
```
(9 × 8) + (2 × 3) + (1 × 2) = 72 + 6 + 2 = 80
```
Now do the same for A and C:
```
user_A = [9, 2, 1]
user_C = [1, 2, 9]

(9 × 1) + (2 × 2) + (1 × 9) = 9 + 4 + 9 = 22
```
80 vs 22. Look at that gap. The operation just told us, using nothing but arithmetic, that A and B agree way more than A and C — with zero knowledge of what "action" or "romance" even mean.

That operation — multiply matching elements, sum the results — is called the **dot product**. It's arguably the single most important operation in all of ML. It's the atomic unit that:

powers recommendation engines (what you just did)
is the core repeated operation inside every matrix multiplication
is literally the mechanism behind attention in transformers (GPT, etc.) — "how much should token A pay attention to token B" is, underneath everything, a dot product
#%% md
**So the real rule:** dot product rewards vectors that are large in the same positions. That's the actual definition of "pointing in a similar direction." If two vectors' big numbers line up on the same axes, the product terms explode. If their big numbers land on different axes, every product term stays small (or goes negative, if the data has negative values — think: opposite sentiment, opposing directions).
#%% md
square each element, sum them, square-root the total. This is called the **L2 norm (or Euclidean norm)**, and it works for a vector of any length — 3 numbers, 300 numbers, 300 million numbers (which is roughly what a modern embedding vector might have). Same formula, every time:
```
||v|| = √```(v₁² + v₂² + v₃² + ... + vₙ²)
```
Now — here's where it all clicks together. We have:

**Dot product** → tells us raw "agreement," but gets inflated by magnitude

**Norm** → tells us the magnitude by itself, with no direction info

What if we took the dot product, and divided out the magnitude contamination — i.e., divided by both vectors' norms multiplied together?
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```
Let's actually run the numbers and see if this fixes our D problem. I'll give you the norms:
```
||A|| = √86 ≈ 9.273
||B|| = √(64+9+4) = √77 ≈ 8.775
||D|| = √(8100+400+100) = √8600 ≈ 92.73   (exactly 10× ||A||, since D = 10×A)
```
So:
```
cos_sim(A, B) = 80 / (9.273 × 8.775) ≈ 80 / 81.37 ≈ 0.983
cos_sim(A, D) = 860 / (9.273 × 92.73) ≈ 860 / 859.85 ≈ 1.000
```
Look at that. cos_sim(A, D) ≈ 1.0 — a perfect match, because A and D have identical direction (D is just A stretched 10x). And cos_sim(A, B) ≈ 0.983 — extremely close but correctly not quite perfect, since B is similar but not identical taste.

Remember the name: cosine similarity. That's not a random name — it's literally the cosine of the angle between the two vectors:
```
cos_sim(A, B) = cos(θ)   where θ = angle between vector A and vector B
```
Think back to trigonometry, angle by angle:
```
θ = 0°   → cos(0°)  = 1    → vectors point in the EXACT same direction
θ = 90°  → cos(90°) = 0    → vectors are PERPENDICULAR (at a right angle)
θ = 180° → cos(180°)= -1   → vectors point in EXACTLY opposite directions
```
- +1 = tastes match perfectly
- 0 = tastes are completely unrelated — knowing one tells you nothing about the other
- -1 = tastes are exact opposites

- If they point in **basically the same direction** → score close to +1
- If they point in t**otally unrelated directions**, like one going right and one going straight up → score is 0
- If they point in **completely opposite directions** → score is -1
"""

### Dot Product
user_A = [9, 2, 1]
user_B = [8, 3, 2]
user_C = [1, 2, 9]

def dot_product_scratch(v1, v2):
    total = 0
    for i in range(len(v1)):
        total += v1[i] * v2[i]   # multiply AND add in the same step
    return total

result = dot_product_scratch(user_A, user_B)
print("Dot Product of A and B:", result)

### Norm of vector
import math

def norm_scratch(v):
    total = 0
    for i in range(len(v)):
        total += v[i]**2          # add the SQUARE of v[i] to total
    return math.sqrt(total)   # square root of the sum

result1 = norm_scratch(user_A)
result2 = norm_scratch(user_B)
result3 = norm_scratch(user_C)
print("Norm of A:", result1)
print("Norm of B:",result2)
print("Norm of C:",result3)

#Cosine similarity
def cosine_similarity_scratch(v1, v2):
    r = dot_product_scratch(v1, v2) / (norm_scratch(v1) * norm_scratch(v2))
    return r

output1 = cosine_similarity_scratch(user_A, user_B)
print("Cosine_Sim_A and B:", output1)
output2 = cosine_similarity_scratch(user_A, user_C)
print("Cosine_Sim_A and C:", output2)

# NumPy version — see np.dot() and np.linalg.norm() collapse everything you just wrote into 2 lines, and understand
# why that matters for speed
import numpy as np

user_A = np.array([9, 2, 1])
user_B = np.array([8, 3, 2])

dot = np.dot(user_A, user_B)
norm_A = np.linalg.norm(user_A)
norm_B = np.linalg.norm(user_B)

# cosine_sim = np.dot(user_A, user_B) / (np.linalg.norm(user_A) * np.linalg.norm(user_B))
cosine_sim = dot / (norm_A * norm_B)
print("Vectorized Numpy version:", cosine_sim)


import numpy as np

users = np.array([
    [9, 2, 1],   # user_A
    [8, 3, 2],   # user_B
    [1, 2, 9],   # user_C
    [90, 20, 10] # user_D
])

user_A = users[0]
similarities = np.dot(users, user_A)
print(similarities)

# One np.dot() call just replaced 4 separate dot-product calls. That's matrix • vector — NumPy takes each row of the
# matrix, dot-products it against the vector, and stacks the 4 results into one output array. No loop written by you
# at all.

# Now — the real power move. What if we want every user compared against every other user, a full similarity table,
# not just A against everyone?
similarity_matrix = np.dot(users, users.T)
print(similarity_matrix)


#Homework_1

matrix = np.array([
    [8, 4, 2],   # user_A
    [6, 4, 2],   # user_B
    [2, 3, 9],   # user_C
    [4, 6, 7] # user_D
])

similarity = np.dot(matrix, matrix.T)
print(similarity)

#Homework_2

norms = np.linalg.norm(matrix, axis=1)
print(norms)

cosine_matrix = similarity / np.outer(norms, norms)
print(cosine_matrix)