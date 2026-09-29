from torch import nn

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class CNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=0, dilation=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, 3, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Output: (batch, 128, 1, 1)

            nn.Flatten(),             # Output: (batch, 128)
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)
