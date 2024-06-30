import matplotlib.pyplot as plt
import numpy as np

def visualize(image, label, prediction, slice_idx):
    
    """
    Visualize a slice of the image, label, and prediction.

    Args:
        image (numpy.ndarray): 3D image array.
        label (numpy.ndarray): 3D label array.
        prediction (numpy.ndarray): 3D prediction array.
        slice_idx (int): Index of the slice to visualize.
    """
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image[slice_idx], cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Label")
    plt.imshow(label[slice_idx])

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(prediction[slice_idx])

    plt.show()
