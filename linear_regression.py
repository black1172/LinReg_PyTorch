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

import torch.nn as nn

# Define a simple linear regression model: y = wx + b
model = nn.Linear(1, 1)  # 1 input feature → 1 output

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass: model prediction
    predictions = model(X)

    # Compute the loss
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Plot data and the model's predictions
predicted = model(X).detach()  # Detach from computation graph for plotting
plt.scatter(X.numpy(), y.numpy(), label="Data")
plt.plot(X.numpy(), predicted.numpy(), color="red", label="Fitted Line")
plt.legend()
plt.title("Model Fit After Training")
plt.show()
