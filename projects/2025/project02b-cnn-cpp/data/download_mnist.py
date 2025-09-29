from torchvision import datasets
import numpy as np
import os

# Create output directory: ../data/mnist/
save_dir = os.path.join(os.path.dirname(__file__), "mnist")
os.makedirs(save_dir, exist_ok=True)

# Download training set
mnist = datasets.MNIST(root=".", train=True, download=True)

# Convert to NumPy arrays
images = mnist.data.numpy()       # shape: (60000, 28, 28)
labels = mnist.targets.numpy()    # shape: (60000,)

# Save as .npy files in ./mnist/
np.save(os.path.join(save_dir, "images.npy"), images)
np.save(os.path.join(save_dir, "labels.npy"), labels)

print(f"Saved images and labels to: {save_dir}")
