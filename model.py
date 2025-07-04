import torch
import torch.nn as nn

class HousePricePredictor(nn.Module):
    def __init__(self):
        super(HousePricePredictor, self).__init__()
        self.linear = nn.Linear(1, 1) # this is our linear layer, this maps our 1 input to our 1 output

    def forward(self, x):
        self.linear(x) # forward is predicting the price of our next element using our linear model
        pass