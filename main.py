import torch
import torch.optim as optim
from model import HousePricePredictor

# Create model
model = HousePricePredictor()

# Scale down both inputs and outputs
sqft = torch.tensor([[1.0], [2.0], [3.0]])    # Instead of 1000, 2000, 3000
prices = torch.tensor([[200.0], [350.0], [500.0]])  # Already scaled

# Use a much smaller learning rate
optimizer = optim.SGD(model.parameters(), lr=0.001)

for j in range(1000):
    for i in range(3):
        # training loop
        prediction = model(sqft[i])  # Predict price for first house
        loss = (prediction - prices[i]) ** 2 # Get our loss
        loss.backward() # Calculate gradients - what direction to adjust
        optimizer.step()
        optimizer.zero_grad()

# After test reports
print("Testing on all houses:")
for i in range(3):
    prediction = model(sqft[i])
    actual = prices[i]
    print(f"House {i+1}: {sqft[i].item():.0f} sqft -> Predicted: ${prediction.item():.2f}k, Actual: ${actual.item():.2f}k")
