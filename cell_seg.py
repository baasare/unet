import os
import glob
import numpy as np
from typing import Tuple, Dict, List

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from PIL import Image


class CellSegmentationModel:
    def __init__(self,
                 img_rows: int = 256,
                 img_cols: int = 256,
                 smooth: float = 1.0):
        """
        Initialize Cell Segmentation U-Net Model

        Args:
            img_rows: Height of input images
            img_cols: Width of input images
            smooth: Smoothing factor for dice coefficient
        """
        K.set_image_data_format('channels_last')
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.smooth = smooth
        self.model = None

    def dice_coef(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute Dice coefficient for model evaluation

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor

        Returns:
            Dice coefficient
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)

    def dice_coef_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Dice coefficient loss function

        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor

        Returns:
            Loss value
        """
        return -self.dice_coef(y_true, y_pred)

    def build_unet(self) -> Model:
        """
        Construct U-Net architecture for cell segmentation

        Returns:
            Compiled Keras model
        """
        inputs = Input((self.img_rows, self.img_cols, 1))

        # Encoder path
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bottom
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        # Decoder path
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        # Output layer
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss=self.dice_coef_loss,
                      metrics=[self.dice_coef])

        self.model = model
        return model

    def load_test_data(self, image_path: str) -> Tuple[List[np.ndarray], Dict[int, str]]:
        """
        Load test images from specified path

        Args:
            image_path: Glob pattern for image files

        Returns:
            Tuple of raw images and filename mapping
        """
        raw_images = []
        image_filename = {}

        for count, filename in enumerate(glob.glob(image_path)):
            try:
                with Image.open(filename) as im:
                    name = os.path.basename(filename)[:-4]
                    im_resized = im.resize((self.img_rows, self.img_cols))
                    raw_images.append(np.array(im_resized))
                    image_filename[count] = name
            except IOError as e:
                print(f'Error loading image {filename}: {e}')

        return raw_images, image_filename

    def preprocess_images(self, imgs: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess images for model input

        Args:
            imgs: List of input images

        Returns:
            Normalized image array
        """
        imgs_p = np.ndarray((len(imgs), self.img_rows, self.img_cols), dtype=np.float32)
        for i, img in enumerate(imgs):
            imgs_p[i] = img.reshape((self.img_rows, self.img_cols)) / 255.0

        imgs_p = imgs_p[..., np.newaxis]

        # Normalize
        mean = imgs_p.mean()
        std = imgs_p.std()
        imgs_p -= mean
        imgs_p /= std

        return imgs_p

    def predict_masks(self,
                      weights_path: str,
                      image_path: str,
                      pred_dir: str) -> None:
        """
        Predict segmentation masks for test images

        Args:
            weights_path: Path to pre-trained model weights
            image_path: Glob pattern for input images
            pred_dir: Directory to save prediction masks
        """
        # Load test data
        raw_images, test_id = self.load_test_data(image_path)
        x_test = self.preprocess_images(raw_images)

        # Build and load model weights
        model = self.build_unet()
        model.load_weights(weights_path)

        # Predict masks
        imgs_mask_predict = model.predict(x_test, verbose=1)
        np.save('imgs_mask_predict.npy', imgs_mask_predict)

        # Create prediction directory if not exists
        os.makedirs(pred_dir, exist_ok=True)

        # Save predictions as images
        for image_pred, index in zip(imgs_mask_predict, range(x_test.shape[0])):
            image_pred = image_pred[:, :, 0]
            image_pred[image_pred > 0.5] *= 255.0
            im = Image.fromarray(image_pred.astype(np.uint8))
            im.save(os.path.join(pred_dir, f"{test_id[index]}_pred.png"))


def main():
    # Configuration
    image_path = 'images/tests/*.png'
    weights_path = 'weights/pre_0_3_5.h5'
    pred_dir = 'images/preds/'

    # Create and run prediction
    segmentation_model = CellSegmentationModel()
    segmentation_model.predict_masks(weights_path, image_path, pred_dir)


if __name__ == '__main__':
    main()