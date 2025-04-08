import numpy as np
u_exact = np.load('u_exact.npy')
print("Min:", np.min(u_exact), "Max:", np.max(u_exact))
print("Contains NaN:", np.isnan(u_exact).any())
print("Contains Inf:", np.isinf(u_exact).any())