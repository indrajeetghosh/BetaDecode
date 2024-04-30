import torch

def apply_augmentation(x, alpha, transforms=[]):
    augmented_x = torch.tensor(x)  # Convert NumPy array to PyTorch tensor
    for transform in transforms:
        augmented_x = transform(augmented_x)  # Apply transformation
    augmented_x += alpha * augmented_x  # Add scaled noise (augmentation)
    return augmented_x