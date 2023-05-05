# Digital Art Protection Algorithm

This Python program provides a way to protect digital art from being used in AI training sets by adding context-aware noise and manipulating spatial frequencies. This makes it harder for AI models to learn meaningful features from the images, while the visual appearance remains nearly the same to human observers.

The program consists of two main functions: `add_context_aware_noise` and `apply_frequency_filter`. These functions are used in the `protect_image` function to process the input image.

## Dependencies

To run this program, you'll need the following libraries:

- OpenCV: For image processing and manipulation.
- NumPy: For working with arrays and numerical operations.

You can install these libraries using pip:

```bash
pip install opencv-python numpy
```

## Functions

### `add_context_aware_noise(image_array, noise_factor)`

This function adds context-aware noise to the input image.

**Arguments:**

- `image_array`: A NumPy array representing the input image.
- `noise_factor`: A float value that determines the strength of the noise.

**Returns:**

- A NumPy array representing the image with added noise.

### `apply_frequency_filter(image_array, filter_factor)`

This function applies a low-pass filter to the input image, which attenuates high-frequency information.

**Arguments:**

- `image_array`: A NumPy array representing the input image.
- `filter_factor`: A float value between 0 and 1 that determines the strength of the filter.

**Returns:**

- A NumPy array representing the filtered image.

### `protect_image(input_image_path, output_image_path, noise_factor, filter_factor)`

This function reads an input image, applies the protection algorithm (adding noise and applying the frequency filter), and saves the processed image.

**Arguments:**

- `input_image_path`: A string representing the path to the input image.
- `output_image_path`: A string representing the path where the processed image should be saved.
- `noise_factor`: A float value that determines the strength of the noise.
- `filter_factor`: A float value between 0 and 1 that determines the strength of the filter.

**Usage Example:**

```python
protect_image("input_image.jpg", "output_image.jpg", noise_factor=0.005, filter_factor=0.5)
```

This will read the image "input_image.jpg", apply the protection algorithm, and save the processed image as "output_image.jpg".

## Limitations and Considerations

This algorithm makes it more difficult for AI models to extract meaningful features from the processed images by introducing noise and reducing high-frequency information. However, it doesn't guarantee that the images will be completely unusable for AI training.

The actual impact on AI training depends on various factors, such as the dataset, model architecture, and training process. This approach aims to increase the difficulty for AI models to learn from the images and could negatively impact their performance.