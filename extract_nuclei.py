import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology, transform

def extract_nuclei(pred_mask_path, min_size=50):
    # Load predicted mask
    pred_mask = io.imread(pred_mask_path)
    pred_mask = pred_mask > 127  # Convert to binary (threshold = 127)

    # Label connected components
    labeled_nuclei = measure.label(pred_mask)

    # Remove small objects (noise)
    # cleaned_nuclei = morphology.remove_small_objects(labeled_nuclei, min_size=min_size)

    return labeled_nuclei

def preprocess_nuclei(segmented_nuclei, output_dir='processed_nuclei'):
    os.makedirs(output_dir, exist_ok=True)

    # Extract individual nuclei from the segmented mask
    for nucleus_label in np.unique(segmented_nuclei):
        if nucleus_label == 0:  # Skip background
            continue

        # Create a binary mask for the current nucleus
        nucleus_mask = (segmented_nuclei == nucleus_label).astype(np.uint8)

        # Resize to 128x128
        nucleus_resized = transform.resize(nucleus_mask, (128, 128), anti_aliasing=True)

        # Save as an image
        io.imsave(f'{output_dir}/nucleus_{nucleus_label}.png', (nucleus_resized * 255).astype(np.uint8))


if __name__ == '__main__':
    # Path to a predicted mask
    pred_mask_path = 'images/preds/img_pred.png'

    # Extract nuclei
    segmented_nuclei = extract_nuclei(pred_mask_path)

    # Visualize the result
    plt.imshow(segmented_nuclei, cmap='nipy_spectral')
    plt.show()

    preprocess_nuclei(segmented_nuclei, output_dir='images/processed_nuclei')
