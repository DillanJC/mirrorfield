"""
Verify SVD singular value ordering.
"""

import numpy as np

# Test with simple matrix
np.random.seed(42)
A = np.random.randn(10, 5)

U, S, Vt = np.linalg.svd(A, full_matrices=False)

print("Singular values from np.linalg.svd:")
print(S)
print()
print(f"S[0] = {S[0]:.4f} (first)")
print(f"S[-1] = {S[-1]:.4f} (last)")
print()
print(f"Is S sorted descending? {np.all(S[:-1] >= S[1:])}")
print()
print("Interpretation:")
print(f"  S[0] = LARGEST singular value")
print(f"  S[-1] = SMALLEST singular value")
print(f"  Ratio S[-1]/S[0] = {S[-1]/S[0]:.6f} (small/large, measures anisotropy)")
