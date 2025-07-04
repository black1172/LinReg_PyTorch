import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_house_data(n_samples=100):
    """Generate fake house price data"""
    # Square footage between 500 and 3000
    sqft = torch.randn(n_samples, 1) * 500 + 1500
    
    # True relationship: price = 150 * sqft + 50000 + some noise
    true_price = 150 * sqft + 50000 + torch.randn(n_samples, 1) * 10000
    
    return sqft, true_price

def plot_data(sqft, prices):
    """Plot the data"""
    plt.figure(figsize=(10, 6))
    plt.scatter(sqft.numpy(), prices.numpy(), alpha=0.7)
    plt.xlabel('Square Footage')
    plt.ylabel('Price ($)')
    plt.title('House Prices vs Square Footage')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Test data generation
    sqft, prices = generate_house_data(100)
    plot_data(sqft, prices)
    print(f"Generated {len(sqft)} house samples")
    print(f"Sqft range: {sqft.min().item():.0f} - {sqft.max().item():.0f}")
    print(f"Price range: ${prices.min().item():.0f} - ${prices.max().item():.0f}")