import torch
import matplotlib.pyplot as plt

# Generate synthetic data: y = 2x + 3 (with some noise)
X = torch.linspace(1, 10, 100).unsqueeze(1)  # shape [100, 1]
y = 2 * X + 3 + torch.randn(X.size()) * 2   # add noise

# Plot the data
plt.scatter(X.numpy(), y.numpy(), label="Data")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (in $1000s)")
plt.title("Synthetic Salary Data")
plt.legend()
plt.show()
