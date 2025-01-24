import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, morphology, exposure, filters, transform


def label_nuclei(pred_mask_path, min_size=50):
    # Load predicted mask
    pred_mask = io.imread(pred_mask_path)

    # Handle grayscale vs binary images
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[:, :, 0]  # Take first channel if multi-channel

    # Use Otsu thresholding for more robust binarization
    thresh = filters.threshold_otsu(pred_mask)
    binary_mask = pred_mask > thresh

    # Label connected components
    labeled_nuclei = measure.label(binary_mask)

    # Remove small objects (noise)
    # cleaned_nuclei = morphology.remove_small_objects(labeled_nuclei, min_size=min_size)

    return labeled_nuclei


def crop_nuclei(segmented_nuclei, output_dir='processed_nuclei', image_name=''):
    os.makedirs(output_dir, exist_ok=True)

    for nucleus_label in np.unique(segmented_nuclei):
        if nucleus_label == 0:  # Skip background
            continue

        # Create a binary mask for the current nucleus
        nucleus_mask = (segmented_nuclei == nucleus_label)

        # Compute bounding box to crop nucleus
        rows = np.any(nucleus_mask, axis=1)
        cols = np.any(nucleus_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the nucleus
        nucleus_crop = nucleus_mask[rmin:rmax + 1, cmin:cmax + 1]

        # Resize to 128x128 with anti-aliasing
        nucleus_resized = transform.resize(nucleus_crop.astype(float),
                                           (128, 128),
                                           anti_aliasing=True,
                                           preserve_range=True)

        # Normalize to improve contrast
        nucleus_norm = exposure.equalize_adapthist(nucleus_resized)

        # Save as an image with original filename context
        io.imsave(f'{output_dir}/{image_name}_nucleus_{nucleus_label}.png',
                  (nucleus_norm * 255).astype(np.uint8),
                  check_contrast=False)


def process_all_predictions(pred_masks_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all prediction mask files
    pred_mask_files = [f for f in os.listdir(pred_masks_dir) if f.endswith('_pred.png')]

    # Process each prediction mask
    for pred_mask_file in pred_mask_files:
        pred_mask_path = os.path.join(pred_masks_dir, pred_mask_file)

        # Label nuclei
        segmented_nuclei = label_nuclei(pred_mask_path)

        # Optional: Visualize the result
        plt.figure()
        plt.title(f'Segmented Nuclei: {pred_mask_file}')
        plt.imshow(segmented_nuclei, cmap='nipy_spectral')
        plt.show()

        # Preprocess, crop and save individual nuclei
        crop_nuclei(segmented_nuclei,
                          output_dir=output_dir,
                          image_name=os.path.splitext(pred_mask_file)[0])


if __name__ == '__main__':
    pred_masks_dir = 'images/preds'
    processed_nuclei_dir = 'images/processed_nuclei'

    # Process all prediction masks
    process_all_predictions(pred_masks_dir, processed_nuclei_dir)